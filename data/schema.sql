-- public.failure_logs definition

-- Drop table

-- DROP TABLE public.failure_logs;

CREATE TABLE public.failure_logs (
	machine_id text NULL,
	"timestamp" timestamp NULL,
	failure_type text NULL
);


-- public.machines definition

-- Drop table

-- DROP TABLE public.machines;

CREATE TABLE public.machines (
	machine_id varchar(10) NOT NULL,
	"name" varchar(100) NULL,
	"type" varchar(50) NULL,
	"location" varchar(100) NULL,
	status varchar(20) DEFAULT 'active'::character varying NULL,
	installation_date timestamp NULL,
	last_maintenance_date timestamp NULL,
	specifications jsonb NULL,
	CONSTRAINT machines_pkey PRIMARY KEY (machine_id)
);


-- public.maintenance_history definition

-- Drop table

-- DROP TABLE public.maintenance_history;

CREATE TABLE public.maintenance_history (
	machine_id text NULL,
	"timestamp" timestamp NULL,
	maintenance_action text NULL
);


-- public.prediction_data definition

-- Drop table

-- DROP TABLE public.prediction_data;

CREATE TABLE public.prediction_data (
	id serial4 NOT NULL,
	machine_id text NULL,
	"timestamp" timestamp NOT NULL,
	afr float8 NULL,
	"current" float8 NULL,
	pressure float8 NULL,
	rpm int4 NULL,
	temperature float8 NULL,
	vibration float8 NULL,
	CONSTRAINT sensor_data_pkey_1 PRIMARY KEY (id),
	CONSTRAINT unique_machine_reading_1 UNIQUE (machine_id, "timestamp")
);
CREATE INDEX idx_sensor_data_machine_id_1 ON public.prediction_data USING btree (machine_id);
CREATE INDEX idx_sensor_data_timestamp_1 ON public.prediction_data USING btree ("timestamp");


-- public.projects definition

-- Drop table

-- DROP TABLE public.projects;

CREATE TABLE public.projects (
	"project name" varchar NOT NULL,
	id bigserial NOT NULL,
	CONSTRAINT projects_pkey PRIMARY KEY (id)
);


-- public."defaults" definition

-- Drop table

-- DROP TABLE public."defaults";

CREATE TABLE public."defaults" (
	id serial4 NOT NULL,
	machine_id varchar(10) NULL,
	category varchar(50) NOT NULL,
	"key" varchar(50) NOT NULL,
	value text NULL,
	description text NULL,
	CONSTRAINT defaults_pkey PRIMARY KEY (id),
	CONSTRAINT unique_machine_category_key UNIQUE (machine_id, category, key),
	CONSTRAINT defaults_machine_id_fkey FOREIGN KEY (machine_id) REFERENCES public.machines(machine_id)
);
CREATE INDEX idx_defaults_machine_id ON public.defaults USING btree (machine_id);


-- public.maintenance definition

-- Drop table

-- DROP TABLE public.maintenance;

CREATE TABLE public.maintenance (
	id serial4 NOT NULL,
	machine_id varchar(10) NULL,
	maintenance_date timestamp NOT NULL,
	completion_date timestamp NULL,
	maintenance_type varchar(50) NOT NULL,
	reason text NOT NULL,
	work_performed text NULL,
	technician_name varchar(100) NULL,
	technician_comments text NULL,
	parts_replaced text NULL,
	status varchar(20) DEFAULT 'completed'::character varying NULL,
	downtime_hours float8 NULL,
	"cost" numeric(10, 2) NULL,
	created_at timestamp DEFAULT CURRENT_TIMESTAMP NULL,
	updated_at timestamp DEFAULT CURRENT_TIMESTAMP NULL,
	CONSTRAINT maintenance_pkey PRIMARY KEY (id),
	CONSTRAINT maintenance_machine_id_fkey FOREIGN KEY (machine_id) REFERENCES public.machines(machine_id)
);
CREATE INDEX idx_maintenance_date ON public.maintenance USING btree (maintenance_date);
CREATE INDEX idx_maintenance_machine_id ON public.maintenance USING btree (machine_id);


-- public.predictions definition

-- Drop table

-- DROP TABLE public.predictions;

CREATE TABLE public.predictions (
	id serial4 NOT NULL,
	machine_id varchar(10) NULL,
	created_at timestamp DEFAULT CURRENT_TIMESTAMP NULL,
	confidence float8 NULL,
	prediction_details jsonb NULL,
	status varchar NOT NULL,
	CONSTRAINT predictions_pkey PRIMARY KEY (id),
	CONSTRAINT predictions_machine_id_fkey FOREIGN KEY (machine_id) REFERENCES public.machines(machine_id)
);
CREATE INDEX idx_predictions_machine_id ON public.predictions USING btree (machine_id);


-- public.sensor_data definition

-- Drop table

-- DROP TABLE public.sensor_data;

CREATE TABLE public.sensor_data (
	id serial4 NOT NULL,
	machine_id text NULL,
	"timestamp" timestamp NOT NULL,
	afr float8 NULL,
	"current" float8 NULL,
	pressure float8 NULL,
	rpm int4 NULL,
	temperature float8 NULL,
	vibration float8 NULL,
	CONSTRAINT sensor_data_pkey PRIMARY KEY (id),
	CONSTRAINT unique_machine_reading UNIQUE (machine_id, "timestamp"),
	CONSTRAINT sensor_data_machine_id_fkey FOREIGN KEY (machine_id) REFERENCES public.machines(machine_id)
);
CREATE INDEX idx_sensor_data_machine_id ON public.sensor_data USING btree (machine_id);
CREATE INDEX idx_sensor_data_timestamp ON public.sensor_data USING btree ("timestamp");


-- public.simulations definition

-- Drop table

-- DROP TABLE public.simulations;

CREATE TABLE public.simulations (
	id serial4 NOT NULL,
	machine_id varchar(10) NULL,
	created_at timestamp DEFAULT CURRENT_TIMESTAMP NULL,
	scenario_type varchar(50) NULL,
	parameters jsonb NULL,
	results jsonb NULL,
	created_by varchar(50) NULL,
	CONSTRAINT simulations_pkey PRIMARY KEY (id),
	CONSTRAINT simulations_machine_id_fkey FOREIGN KEY (machine_id) REFERENCES public.machines(machine_id)
);
CREATE INDEX idx_simulations_machine_id ON public.simulations USING btree (machine_id);