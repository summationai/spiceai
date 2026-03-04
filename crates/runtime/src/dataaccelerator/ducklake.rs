/*
Copyright 2026 The Spice.ai OSS Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

use super::{AccelerationSource, BootstrapStatus, DataAccelerator};
use crate::{
    App, Runtime,
    component::dataset::acceleration::{Acceleration, Engine, Mode},
    dataaccelerator::{
        FilePathError,
        duckdb::{DEFAULT_MIN_IDLE_CONNECTIONS, create_table_provider, settings},
    },
    datafusion::{dialect::new_duckdb_dialect, udf::deny_spice_specific_functions},
    make_spice_data_directory,
    parameters::ParameterSpec,
    register_data_accelerator, spice_data_base_path,
};
use arrow::datatypes::DataType;
use arrow_flight::error::FlightError;
use async_trait::async_trait;
use data_components::flight::{FlightTable, write::FlightTableWriter};
use datafusion::{
    common::DFSchemaRef, datasource::TableProvider, logical_expr::CreateExternalTable,
    sql::TableReference,
};
use datafusion_table_providers::{
    duckdb::{DuckDBSettingsRegistry, DuckDBTableProviderFactory},
    sql::db_connection_pool::duckdbpool::{DuckDbConnectionPool, DuckDbConnectionPoolBuilder},
};
use duckdb::AccessMode;
use flight_client::{Credentials, FlightClient};
use futures::StreamExt;
use runtime_table_partition::expression::PartitionedBy;
use secrecy::SecretString;
use settings::OrderByNonIntegerLiteral;
use snafu::prelude::*;
use std::{any::Any, cmp::max, collections::HashMap, sync::Arc};

#[derive(Debug, Snafu)]
pub enum Error {
    #[snafu(display("Acceleration creation failed: {source}"))]
    AccelerationCreationFailed {
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[snafu(display("Acceleration initialization failed: {source}"))]
    AccelerationInitializationFailed {
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[snafu(display("Acceleration not enabled for dataset: {dataset}"))]
    AccelerationNotEnabled { dataset: Arc<str> },

    #[snafu(display("Invalid DuckLake acceleration configuration: {detail}"))]
    InvalidConfiguration { detail: Arc<str> },

    #[snafu(display(
        "Missing required DuckLake acceleration parameters. Specify either ducklake_connection_string, or ducklake_endpoint + ducklake_username + ducklake_password"
    ))]
    MissingConnectionConfiguration,

    #[snafu(display("Missing required DuckLake acceleration parameter: ducklake_username"))]
    MissingEndpointUsername,

    #[snafu(display("Missing required DuckLake acceleration parameter: ducklake_password"))]
    MissingEndpointPassword,

    #[snafu(display("Unable to create DuckLake endpoint Flight client: {source}"))]
    UnableToCreateFlightClient { source: flight_client::Error },

    #[snafu(display("Unable to execute DuckLake endpoint SQL: {source}"))]
    UnableToExecuteRemoteSql { source: flight_client::Error },

    #[snafu(display("DuckLake endpoint SQL stream failed: {source}"))]
    RemoteSqlStream { source: FlightError },

    #[snafu(display("Unable to create DuckLake endpoint table provider: {source}"))]
    UnableToCreateRemoteTableProvider {
        source: Box<dyn std::error::Error + Send + Sync>,
    },
}

type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Clone, Debug)]
struct EndpointConfig {
    endpoint: String,
    username: String,
    password: SecretString,
}

pub struct DuckLakeAccelerator {
    duckdb_factory: DuckDBTableProviderFactory,
}

impl DuckLakeAccelerator {
    #[must_use]
    pub fn new() -> Self {
        Self {
            duckdb_factory: DuckDBTableProviderFactory::new(AccessMode::ReadWrite)
                .with_dialect(new_duckdb_dialect())
                .with_settings_registry(
                    DuckDBSettingsRegistry::new()
                        .with_setting(Box::new(OrderByNonIntegerLiteral))
                        .with_setting(Box::new(settings::IndexScanPercentage))
                        .with_setting(Box::new(settings::IndexScanMaxCount))
                        .with_setting(Box::new(settings::TimeZone)),
                )
                .with_function_support(deny_spice_specific_functions()),
        }
    }

    fn non_empty_param<'a>(params: &'a HashMap<String, String>, keys: &[&str]) -> Option<&'a str> {
        keys.iter().find_map(|key| {
            params
                .get(*key)
                .map(String::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
        })
    }

    pub fn ducklake_open_path(&self, source: &dyn AccelerationSource) -> Result<String> {
        if !source.is_file_accelerated() {
            return Err(Error::InvalidConfiguration {
                detail: Arc::from("Dataset is not file accelerated"),
            });
        }

        let Some(acceleration) = source.acceleration() else {
            unreachable!("Expected source acceleration settings to exist")
        };

        if let Some(path) = Self::non_empty_param(&acceleration.params, &["open", "ducklake_open"])
        {
            return Ok(path.to_string());
        }

        Ok(format!(
            "{}/accelerated_ducklake.db",
            spice_data_base_path()
        ))
    }

    fn ducklake_connection_string(source: &dyn AccelerationSource) -> Option<String> {
        source.acceleration().and_then(|acceleration| {
            Self::non_empty_param(
                &acceleration.params,
                &["connection_string", "ducklake_connection_string"],
            )
            .map(ToString::to_string)
        })
    }

    fn ducklake_endpoint(source: &dyn AccelerationSource) -> Option<String> {
        source.acceleration().and_then(|acceleration| {
            Self::non_empty_param(&acceleration.params, &["endpoint", "ducklake_endpoint"])
                .map(ToString::to_string)
        })
    }

    fn ducklake_catalog_name(source: &dyn AccelerationSource) -> String {
        source
            .acceleration()
            .and_then(|acceleration| {
                Self::non_empty_param(
                    &acceleration.params,
                    &["catalog_name", "ducklake_catalog_name"],
                )
            })
            .map(ToString::to_string)
            .unwrap_or_else(|| "ducklake".to_string())
    }

    fn endpoint_config_from_options(
        options: &HashMap<String, String>,
    ) -> Result<Option<EndpointConfig>> {
        let Some(endpoint) = Self::non_empty_param(options, &["endpoint", "ducklake_endpoint"])
        else {
            return Ok(None);
        };

        let username = Self::non_empty_param(options, &["username", "user", "ducklake_username"])
            .map(ToString::to_string)
            .context(MissingEndpointUsernameSnafu)?;

        let password = Self::non_empty_param(options, &["password", "pass", "ducklake_password"])
            .map(|value| SecretString::new(value.to_string().into()))
            .context(MissingEndpointPasswordSnafu)?;

        Ok(Some(EndpointConfig {
            endpoint: endpoint.to_string(),
            username,
            password,
        }))
    }

    fn normalize_flight_endpoint(endpoint: &str) -> String {
        if endpoint.starts_with("grpc+tls://")
            || endpoint.starts_with("grpc://")
            || endpoint.starts_with("https://")
            || endpoint.starts_with("http://")
        {
            endpoint.to_string()
        } else {
            format!("grpc+tls://{endpoint}")
        }
    }

    fn quote_identifier(identifier: &str) -> String {
        format!("\"{}\"", identifier.replace('"', "\"\""))
    }

    fn quote_table_reference(table_reference: &TableReference) -> String {
        match table_reference {
            TableReference::Bare { table } => Self::quote_identifier(table),
            TableReference::Partial { schema, table } => format!(
                "{}.{}",
                Self::quote_identifier(schema),
                Self::quote_identifier(table)
            ),
            TableReference::Full {
                catalog,
                schema,
                table,
            } => format!(
                "{}.{}.{}",
                Self::quote_identifier(catalog),
                Self::quote_identifier(schema),
                Self::quote_identifier(table)
            ),
        }
    }

    fn to_ducklake_sql_type(data_type: &DataType) -> String {
        match data_type {
            DataType::Int8 => "TINYINT".to_string(),
            DataType::Int16 => "SMALLINT".to_string(),
            DataType::Int32 => "INTEGER".to_string(),
            DataType::Int64 => "BIGINT".to_string(),
            DataType::UInt8 => "UTINYINT".to_string(),
            DataType::UInt16 => "USMALLINT".to_string(),
            DataType::UInt32 => "UINTEGER".to_string(),
            DataType::UInt64 => "UBIGINT".to_string(),
            DataType::Float32 => "FLOAT".to_string(),
            DataType::Float64 => "DOUBLE".to_string(),
            DataType::Utf8 | DataType::LargeUtf8 => "VARCHAR".to_string(),
            DataType::Binary | DataType::LargeBinary | DataType::FixedSizeBinary(_) => {
                "BLOB".to_string()
            }
            DataType::Boolean => "BOOLEAN".to_string(),
            DataType::Date32 => "DATE".to_string(),
            DataType::Date64 => "TIMESTAMP".to_string(),
            DataType::Time32(_) | DataType::Time64(_) => "TIME".to_string(),
            DataType::Timestamp(_, tz) => {
                if tz.is_some() {
                    "TIMESTAMPTZ".to_string()
                } else {
                    "TIMESTAMP".to_string()
                }
            }
            DataType::Decimal128(precision, scale) => format!("DECIMAL({precision}, {scale})"),
            DataType::Duration(_) => "BIGINT".to_string(),
            DataType::Interval(_) => "INTERVAL".to_string(),
            // Complex and less common scalar types are serialized to JSON/text for the PoC path.
            _ => "VARCHAR".to_string(),
        }
    }

    fn build_create_table_sql(table_reference: &TableReference, schema: &DFSchemaRef) -> String {
        let table_name = Self::quote_table_reference(table_reference);
        let columns = schema
            .fields()
            .iter()
            .map(|field| {
                let column_name = Self::quote_identifier(field.name());
                let column_type = Self::to_ducklake_sql_type(field.data_type());
                let nullability = if field.is_nullable() { "" } else { " NOT NULL" };
                format!("{column_name} {column_type}{nullability}")
            })
            .collect::<Vec<_>>()
            .join(", ");

        format!("CREATE TABLE IF NOT EXISTS {table_name} ({columns})")
    }

    async fn execute_remote_sql(flight_client: &mut FlightClient, sql: &str) -> Result<()> {
        let mut stream = flight_client
            .query(sql)
            .await
            .context(UnableToExecuteRemoteSqlSnafu)?;

        while let Some(batch) = stream.next().await {
            batch.context(RemoteSqlStreamSnafu)?;
        }

        Ok(())
    }

    async fn create_remote_table_provider(
        &self,
        cmd: &CreateExternalTable,
        endpoint_config: EndpointConfig,
    ) -> Result<Arc<dyn TableProvider>, Box<dyn std::error::Error + Send + Sync>> {
        let endpoint = Self::normalize_flight_endpoint(&endpoint_config.endpoint);
        let credentials = Credentials::new(&endpoint_config.username, endpoint_config.password);

        let mut flight_client = FlightClient::try_new(Arc::from(endpoint), credentials, None, None)
            .await
            .context(UnableToCreateFlightClientSnafu)
            .boxed()?;

        let create_table_sql = Self::build_create_table_sql(&cmd.name, &cmd.schema);
        if let Err(err) = Self::execute_remote_sql(&mut flight_client, &create_table_sql).await {
            let err_text = err.to_string();
            // GizmoSQL can execute DDL successfully but still return "Unable to parse command"
            // when no result stream is produced for the statement.
            if err_text.contains("Unable to parse command") {
                tracing::warn!(
                    "Ignoring DuckLake endpoint parse error for DDL command and continuing: {err_text}"
                );
            } else {
                return Err(Box::new(err));
            }
        }

        // GizmoSQL's Flight endpoint may not implement get_schema/get_query_schema fully.
        // Build the read provider with the already known CreateExternalTable schema to skip
        // schema introspection RPCs during initialization.
        let read_provider = Arc::new(
            Arc::new(FlightTable::create_with_schema(
                "ducklake",
                flight_client.clone(),
                cmd.name.clone(),
                Arc::new(cmd.schema.as_arrow().clone()),
                new_duckdb_dialect(),
                None,
            ))
            .create_federated_table_provider(),
        );

        Ok(FlightTableWriter::create(
            read_provider,
            cmd.name.clone(),
            flight_client,
        ))
    }

    fn with_ducklake_setup_queries(
        mut pool_builder: DuckDbConnectionPoolBuilder,
        connection_string: &str,
        catalog_name: &str,
    ) -> DuckDbConnectionPoolBuilder {
        let escaped_connection_string = connection_string.replace('\'', "''");
        let escaped_catalog_name = catalog_name.replace('"', "\"\"");
        let attach_sql =
            format!("ATTACH 'ducklake:{escaped_connection_string}' AS \"{escaped_catalog_name}\"");
        let use_sql = format!("USE \"{escaped_catalog_name}\"");

        pool_builder =
            pool_builder.with_connection_setup_query("PRAGMA enable_checkpoint_on_shutdown");
        pool_builder = pool_builder.with_connection_setup_query("INSTALL ducklake");
        pool_builder = pool_builder.with_connection_setup_query("LOAD ducklake");
        pool_builder = pool_builder.with_connection_setup_query(attach_sql);
        pool_builder.with_connection_setup_query(use_sql)
    }

    /// Returns an existing DuckDB connection pool configured for DuckLake, or creates a new one.
    ///
    /// For endpoint mode (`ducklake_endpoint`), this pool is used for local checkpoint/state
    /// metadata while table reads/writes are performed against the remote Flight endpoint.
    pub async fn get_shared_pool(
        &self,
        source: &dyn AccelerationSource,
    ) -> Result<DuckDbConnectionPool> {
        let acceleration = source.acceleration().context(AccelerationNotEnabledSnafu {
            dataset: source.name().to_string(),
        })?;

        let connection_string = Self::ducklake_connection_string(source);
        let endpoint = Self::ducklake_endpoint(source);
        ensure!(
            connection_string.is_some() || endpoint.is_some(),
            MissingConnectionConfigurationSnafu
        );

        let catalog_name = Self::ducklake_catalog_name(source);

        let pool = match acceleration.mode {
            Mode::File | Mode::FileCreate => {
                let open_path = self.ducklake_open_path(source)?;
                let num_accelerating_datasets = self.get_num_accelerating_datasets(
                    Some(open_path.as_str()),
                    &source.app(),
                    source.runtime(),
                );
                let max_size = Self::get_pool_max_size(num_accelerating_datasets, acceleration);
                let pool_builder = DuckDbConnectionPoolBuilder::file(&open_path)
                    .with_max_size(Some(max_size))
                    .with_min_idle(Some(DEFAULT_MIN_IDLE_CONNECTIONS));

                let pool_builder = if let Some(connection_string) = connection_string.as_deref() {
                    Self::with_ducklake_setup_queries(
                        pool_builder,
                        connection_string,
                        &catalog_name,
                    )
                } else {
                    pool_builder.with_connection_setup_query("PRAGMA enable_checkpoint_on_shutdown")
                };

                self.duckdb_factory
                    .get_or_init_instance_with_builder(pool_builder)
                    .await
                    .boxed()
                    .context(AccelerationCreationFailedSnafu)?
            }
            Mode::Memory => {
                let num_accelerating_datasets =
                    self.get_num_accelerating_datasets(None, &source.app(), source.runtime());
                let max_size = Self::get_pool_max_size(num_accelerating_datasets, acceleration);
                let pool_builder = DuckDbConnectionPoolBuilder::memory()
                    .with_max_size(Some(max_size))
                    .with_min_idle(Some(DEFAULT_MIN_IDLE_CONNECTIONS));

                let pool_builder = if let Some(connection_string) = connection_string.as_deref() {
                    Self::with_ducklake_setup_queries(
                        pool_builder,
                        connection_string,
                        &catalog_name,
                    )
                } else {
                    pool_builder.with_connection_setup_query("PRAGMA enable_checkpoint_on_shutdown")
                };

                self.duckdb_factory
                    .get_or_init_instance_with_builder(pool_builder)
                    .await
                    .boxed()
                    .context(AccelerationCreationFailedSnafu)?
            }
        };

        Ok(pool)
    }

    fn get_num_accelerating_datasets(
        &self,
        path: Option<&str>,
        app: &Arc<App>,
        rt: Arc<Runtime>,
    ) -> u32 {
        let mut instance_usage: u32 = 1;

        let datasets = rt.get_valid_datasets(app, crate::LogErrors(false));
        for ds in datasets {
            if let Some(acceleration) = &ds.acceleration {
                if acceleration.engine != Engine::DuckLake {
                    continue;
                }

                if let Some(this_file_path) = path {
                    if matches!(acceleration.mode, Mode::File | Mode::FileCreate)
                        && let Ok(file_path) = self.file_path(ds.as_ref())
                        && this_file_path == file_path
                    {
                        instance_usage += 1;
                    }
                } else if acceleration.mode == Mode::Memory {
                    instance_usage += 1;
                }
            }
        }

        instance_usage
    }

    fn get_pool_max_size(num_accelerating_datasets: u32, acceleration: &Acceleration) -> u32 {
        let pool_size_param = acceleration
            .params
            .get("connection_pool_size")
            .and_then(|size_str| size_str.parse::<u32>().ok());

        pool_size_param
            .unwrap_or_else(|| max(DEFAULT_MIN_IDLE_CONNECTIONS, num_accelerating_datasets))
    }
}

impl Default for DuckLakeAccelerator {
    fn default() -> Self {
        Self::new()
    }
}

const PARAMETERS: &[ParameterSpec] = &[
    ParameterSpec::runtime("file_watcher"),
    ParameterSpec::component("open").description(
        "Optional path to an existing DuckDB file used to host local DuckLake metadata/checkpoint state.",
    ),
    ParameterSpec::component("connection_string").description(
        "DuckLake metadata connection string (for example, s3://bucket/path/metadata.ducklake).",
    ),
    ParameterSpec::component("catalog_name")
        .description("The attached DuckLake catalog name in DuckDB. Defaults to 'ducklake'."),
    ParameterSpec::component("endpoint").description(
        "Optional hosted DuckLake Flight endpoint (for example, appexperiment-shared-ducklake-1-v2.summation.com:443 or grpc+tls://...).",
    ),
    ParameterSpec::component("username")
        .description("Username for hosted DuckLake endpoint authentication."),
    ParameterSpec::component("password")
        .secret()
        .description("Password for hosted DuckLake endpoint authentication."),
    ParameterSpec::runtime("connection_pool_size").description(
        "The maximum number of client connections created in the ducklake connection pool.",
    ),
    ParameterSpec::runtime("on_refresh_recompute_statistics"),
    ParameterSpec::runtime("optimizer_duckdb_aggregate_pushdown"),
];

#[async_trait]
impl DataAccelerator for DuckLakeAccelerator {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &'static str {
        "ducklake"
    }

    fn valid_file_extensions(&self) -> Vec<&'static str> {
        vec!["db", "ddb", "duckdb"]
    }

    fn file_path(&self, source: &dyn AccelerationSource) -> Result<String, FilePathError> {
        self.ducklake_open_path(source)
            .map_err(|e| FilePathError::External {
                engine: Engine::DuckLake,
                source: e.into(),
            })
    }

    fn is_initialized(&self, source: &dyn AccelerationSource) -> bool {
        if !source.is_file_accelerated() {
            return true;
        }
        self.has_existing_file(source)
    }

    async fn init(
        &self,
        source: &dyn AccelerationSource,
    ) -> Result<BootstrapStatus, Box<dyn std::error::Error + Send + Sync>> {
        if !source.is_file_accelerated() {
            self.get_shared_pool(source).await?;
            return Ok(BootstrapStatus::none());
        }

        let path = self.file_path(source)?;

        if let Some(acceleration) = source.acceleration() {
            if !acceleration.params.contains_key("open")
                && !acceleration.params.contains_key("ducklake_open")
            {
                make_spice_data_directory().map_err(|err| {
                    Error::AccelerationInitializationFailed { source: err.into() }
                })?;
            }

            if acceleration.mode == Mode::FileCreate {
                let file_path = std::path::Path::new(&path);
                if file_path.exists() {
                    tracing::warn!(
                        "DuckLake acceleration mode is 'file_create', removing existing file: {}",
                        path
                    );
                    std::fs::remove_file(file_path).map_err(|err| {
                        Error::AccelerationInitializationFailed { source: err.into() }
                    })?;
                }
            }
        }

        self.get_shared_pool(source).await?;
        Ok(BootstrapStatus::none())
    }

    async fn create_external_table(
        &self,
        mut cmd: CreateExternalTable,
        source: Option<&dyn AccelerationSource>,
        _partition_by: Vec<PartitionedBy>,
    ) -> Result<Arc<dyn TableProvider>, Box<dyn std::error::Error + Send + Sync>> {
        if let Some(duckdb_file) = cmd.options.remove("file") {
            cmd.options.insert("open".to_string(), duckdb_file);
        }

        if let Some(ducklake_open) = cmd.options.remove("ducklake_open") {
            cmd.options.insert("open".to_string(), ducklake_open);
        }

        if let Some(recompute_statistics_on_write) =
            cmd.options.remove("on_refresh_recompute_statistics")
        {
            cmd.options.insert(
                "recompute_statistics_on_write".to_string(),
                recompute_statistics_on_write,
            );
        }

        if let Some(source) = source {
            if let Some(temp_directory) = source
                .app()
                .runtime
                .query
                .clone()
                .unwrap_or_default()
                .temp_directory
            {
                cmd.options
                    .insert("temp_directory".to_string(), temp_directory);
            }

            if source.is_file_accelerated() && !cmd.options.contains_key("open") {
                let open_path = self.ducklake_open_path(source)?;
                cmd.options.insert("open".to_string(), open_path);
            }

            // Ensure local metadata/checkpoint pool is initialized for this source.
            self.get_shared_pool(source).await?;
        }

        if let Some(endpoint_config) = Self::endpoint_config_from_options(&cmd.options)? {
            return self
                .create_remote_table_provider(&cmd, endpoint_config)
                .await;
        }

        Ok(create_table_provider(&self.duckdb_factory, &cmd, None).await?)
    }

    fn prefix(&self) -> &'static str {
        "ducklake"
    }

    fn parameters(&self) -> &'static [ParameterSpec] {
        PARAMETERS
    }
}

register_data_accelerator!(Engine::DuckLake, DuckLakeAccelerator);
