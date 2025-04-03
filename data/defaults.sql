-- Populate defaults table with system defaults 

-- Machine-specific sensor thresholds

-- M001: Precision milling machine
INSERT INTO defaults (machine_id, category, key, value, description)
VALUES 
    ('M001', 'sensor_thresholds', 'afr_min', '10.5', 'M001-specific minimum AFR'),
    ('M001', 'sensor_thresholds', 'afr_max', '14.5', 'M001-specific maximum AFR'),
    ('M001', 'sensor_thresholds', 'rpm_min', '3000', 'M001-specific minimum RPM'),
    ('M001', 'sensor_thresholds', 'rpm_max', '3400', 'M001-specific maximum RPM'),
    ('M001', 'sensor_thresholds', 'current_min', '21.0', 'M001-specific minimum current'),
    ('M001', 'sensor_thresholds', 'current_max', '39.0', 'M001-specific maximum current'),
    ('M001', 'sensor_thresholds', 'pressure_min', '3.2', 'M001-specific minimum pressure'),
    ('M001', 'sensor_thresholds', 'pressure_max', '6.8', 'M001-specific maximum pressure'),
    ('M001', 'sensor_thresholds', 'temperature_min', '60.0', 'M001-specific minimum temperature'),
    ('M001', 'sensor_thresholds', 'temperature_max', '85.0', 'M001-specific maximum temperature'),
    ('M001', 'sensor_thresholds', 'vibration_max', '5.0', 'M001-specific maximum vibration');

-- M002: Heavy duty lathe
INSERT INTO defaults (machine_id, category, key, value, description)
VALUES 
    ('M002', 'sensor_thresholds', 'afr_min', '9.8', 'M002-specific minimum AFR'),
    ('M002', 'sensor_thresholds', 'afr_max', '15.5', 'M002-specific maximum AFR'),
    ('M002', 'sensor_thresholds', 'rpm_min', '2700', 'M002-specific minimum RPM'),
    ('M002', 'sensor_thresholds', 'rpm_max', '3700', 'M002-specific maximum RPM'),
    ('M002', 'sensor_thresholds', 'current_min', '25.0', 'M002-specific minimum current'),
    ('M002', 'sensor_thresholds', 'current_max', '45.0', 'M002-specific maximum current'),
    ('M002', 'sensor_thresholds', 'pressure_min', '3.5', 'M002-specific minimum pressure'),
    ('M002', 'sensor_thresholds', 'pressure_max', '7.5', 'M002-specific maximum pressure'),
    ('M002', 'sensor_thresholds', 'temperature_min', '70.0', 'M002-specific minimum temperature'),
    ('M002', 'sensor_thresholds', 'temperature_max', '105.0', 'M002-specific maximum temperature'),
    ('M002', 'sensor_thresholds', 'vibration_max', '9.0', 'M002-specific maximum vibration');

-- M003: Precision assembly robot
INSERT INTO defaults (machine_id, category, key, value, description)
VALUES 
    ('M003', 'sensor_thresholds', 'afr_min', '11.0', 'M003-specific minimum AFR'),
    ('M003', 'sensor_thresholds', 'afr_max', '14.0', 'M003-specific maximum AFR'),
    ('M003', 'sensor_thresholds', 'rpm_min', '3200', 'M003-specific minimum RPM'),
    ('M003', 'sensor_thresholds', 'rpm_max', '3400', 'M003-specific maximum RPM'),
    ('M003', 'sensor_thresholds', 'current_min', '19.0', 'M003-specific minimum current'),
    ('M003', 'sensor_thresholds', 'current_max', '37.0', 'M003-specific maximum current'),
    ('M003', 'sensor_thresholds', 'pressure_min', '3.5', 'M003-specific minimum pressure'),
    ('M003', 'sensor_thresholds', 'pressure_max', '6.0', 'M003-specific maximum pressure'),
    ('M003', 'sensor_thresholds', 'temperature_min', '62.0', 'M003-specific minimum temperature'),
    ('M003', 'sensor_thresholds', 'temperature_max', '90.0', 'M003-specific maximum temperature'),
    ('M003', 'sensor_thresholds', 'vibration_max', '3.0', 'M003-specific maximum vibration');

-- M004: Injection molding machine
INSERT INTO defaults (machine_id, category, key, value, description)
VALUES 
    ('M004', 'sensor_thresholds', 'afr_min', '10.2', 'M004-specific minimum AFR'),
    ('M004', 'sensor_thresholds', 'afr_max', '15.2', 'M004-specific maximum AFR'),
    ('M004', 'sensor_thresholds', 'rpm_min', '2900', 'M004-specific minimum RPM'),
    ('M004', 'sensor_thresholds', 'rpm_max', '3500', 'M004-specific maximum RPM'),
    ('M004', 'sensor_thresholds', 'current_min', '22.0', 'M004-specific minimum current'),
    ('M004', 'sensor_thresholds', 'current_max', '42.0', 'M004-specific maximum current'),
    ('M004', 'sensor_thresholds', 'pressure_min', '4.5', 'M004-specific minimum pressure'),
    ('M004', 'sensor_thresholds', 'pressure_max', '8.5', 'M004-specific maximum pressure'),
    ('M004', 'sensor_thresholds', 'temperature_min', '75.0', 'M004-specific minimum temperature'),
    ('M004', 'sensor_thresholds', 'temperature_max', '110.0', 'M004-specific maximum temperature'),
    ('M004', 'sensor_thresholds', 'vibration_max', '7.5', 'M004-specific maximum vibration');

-- M005: CNC router
INSERT INTO defaults (machine_id, category, key, value, description)
VALUES 
    ('M005', 'sensor_thresholds', 'afr_min', '10.8', 'M005-specific minimum AFR'),
    ('M005', 'sensor_thresholds', 'afr_max', '14.3', 'M005-specific maximum AFR'),
    ('M005', 'sensor_thresholds', 'rpm_min', '3500', 'M005-specific minimum RPM'),
    ('M005', 'sensor_thresholds', 'rpm_max', '4200', 'M005-specific maximum RPM'),
    ('M005', 'sensor_thresholds', 'current_min', '22.0', 'M005-specific minimum current'),
    ('M005', 'sensor_thresholds', 'current_max', '36.0', 'M005-specific maximum current'),
    ('M005', 'sensor_thresholds', 'pressure_min', '2.8', 'M005-specific minimum pressure'),
    ('M005', 'sensor_thresholds', 'pressure_max', '6.5', 'M005-specific maximum pressure'),
    ('M005', 'sensor_thresholds', 'temperature_min', '63.0', 'M005-specific minimum temperature'),
    ('M005', 'sensor_thresholds', 'temperature_max', '92.0', 'M005-specific maximum temperature'),
    ('M005', 'sensor_thresholds', 'vibration_max', '6.0', 'M005-specific maximum vibration');

-- M006: Industrial press
INSERT INTO defaults (machine_id, category, key, value, description)
VALUES 
    ('M006', 'sensor_thresholds', 'afr_min', '9.5', 'M006-specific minimum AFR'),
    ('M006', 'sensor_thresholds', 'afr_max', '16.0', 'M006-specific maximum AFR'),
    ('M006', 'sensor_thresholds', 'rpm_min', '2600', 'M006-specific minimum RPM'),
    ('M006', 'sensor_thresholds', 'rpm_max', '3800', 'M006-specific maximum RPM'),
    ('M006', 'sensor_thresholds', 'current_min', '30.0', 'M006-specific minimum current'),
    ('M006', 'sensor_thresholds', 'current_max', '50.0', 'M006-specific maximum current'),
    ('M006', 'sensor_thresholds', 'pressure_min', '5.0', 'M006-specific minimum pressure'),
    ('M006', 'sensor_thresholds', 'pressure_max', '9.0', 'M006-specific maximum pressure'),
    ('M006', 'sensor_thresholds', 'temperature_min', '68.0', 'M006-specific minimum temperature'),
    ('M006', 'sensor_thresholds', 'temperature_max', '100.0', 'M006-specific maximum temperature'),
    ('M006', 'sensor_thresholds', 'vibration_max', '9.5', 'M006-specific maximum vibration');

-- M007: Automated packaging machine
INSERT INTO defaults (machine_id, category, key, value, description)
VALUES 
    ('M007', 'sensor_thresholds', 'afr_min', '11.5', 'M007-specific minimum AFR'),
    ('M007', 'sensor_thresholds', 'afr_max', '13.5', 'M007-specific maximum AFR'),
    ('M007', 'sensor_thresholds', 'rpm_min', '2600', 'M007-specific minimum RPM'),
    ('M007', 'sensor_thresholds', 'rpm_max', '3200', 'M007-specific maximum RPM'),
    ('M007', 'sensor_thresholds', 'current_min', '18.0', 'M007-specific minimum current'),
    ('M007', 'sensor_thresholds', 'current_max', '35.0', 'M007-specific maximum current'),
    ('M007', 'sensor_thresholds', 'pressure_min', '2.5', 'M007-specific minimum pressure'),
    ('M007', 'sensor_thresholds', 'pressure_max', '5.5', 'M007-specific maximum pressure'),
    ('M007', 'sensor_thresholds', 'temperature_min', '60.0', 'M007-specific minimum temperature'),
    ('M007', 'sensor_thresholds', 'temperature_max', '88.0', 'M007-specific maximum temperature'),
    ('M007', 'sensor_thresholds', 'vibration_max', '5.5', 'M007-specific maximum vibration');

-- M008: Welding robot
INSERT INTO defaults (machine_id, category, key, value, description)
VALUES 
    ('M008', 'sensor_thresholds', 'afr_min', '9.0', 'M008-specific minimum AFR'),
    ('M008', 'sensor_thresholds', 'afr_max', '16.5', 'M008-specific maximum AFR'),
    ('M008', 'sensor_thresholds', 'rpm_min', '3000', 'M008-specific minimum RPM'),
    ('M008', 'sensor_thresholds', 'rpm_max', '3800', 'M008-specific maximum RPM'),
    ('M008', 'sensor_thresholds', 'current_min', '35.0', 'M008-specific minimum current'),
    ('M008', 'sensor_thresholds', 'current_max', '55.0', 'M008-specific maximum current'),
    ('M008', 'sensor_thresholds', 'pressure_min', '3.8', 'M008-specific minimum pressure'),
    ('M008', 'sensor_thresholds', 'pressure_max', '7.2', 'M008-specific maximum pressure'),
    ('M008', 'sensor_thresholds', 'temperature_min', '80.0', 'M008-specific minimum temperature'),
    ('M008', 'sensor_thresholds', 'temperature_max', '120.0', 'M008-specific maximum temperature'),
    ('M008', 'sensor_thresholds', 'vibration_max', '8.0', 'M008-specific maximum vibration');

-- M009: Conveyor system
INSERT INTO defaults (machine_id, category, key, value, description)
VALUES 
    ('M009', 'sensor_thresholds', 'afr_min', '10.0', 'M009-specific minimum AFR'),
    ('M009', 'sensor_thresholds', 'afr_max', '14.0', 'M009-specific maximum AFR'),
    ('M009', 'sensor_thresholds', 'rpm_min', '1800', 'M009-specific minimum RPM'),
    ('M009', 'sensor_thresholds', 'rpm_max', '2400', 'M009-specific maximum RPM'),
    ('M009', 'sensor_thresholds', 'current_min', '15.0', 'M009-specific minimum current'),
    ('M009', 'sensor_thresholds', 'current_max', '30.0', 'M009-specific maximum current'),
    ('M009', 'sensor_thresholds', 'pressure_min', '2.0', 'M009-specific minimum pressure'),
    ('M009', 'sensor_thresholds', 'pressure_max', '5.0', 'M009-specific maximum pressure'),
    ('M009', 'sensor_thresholds', 'temperature_min', '55.0', 'M009-specific minimum temperature'),
    ('M009', 'sensor_thresholds', 'temperature_max', '85.0', 'M009-specific maximum temperature'),
    ('M009', 'sensor_thresholds', 'vibration_max', '4.0', 'M009-specific maximum vibration');

-- M010: Quality control station
INSERT INTO defaults (machine_id, category, key, value, description)
VALUES 
    ('M010', 'sensor_thresholds', 'afr_min', '12.0', 'M010-specific minimum AFR'),
    ('M010', 'sensor_thresholds', 'afr_max', '13.0', 'M010-specific maximum AFR'),
    ('M010', 'sensor_thresholds', 'rpm_min', '3100', 'M010-specific minimum RPM'),
    ('M010', 'sensor_thresholds', 'rpm_max', '3300', 'M010-specific maximum RPM'),
    ('M010', 'sensor_thresholds', 'current_min', '18.5', 'M010-specific minimum current'),
    ('M010', 'sensor_thresholds', 'current_max', '32.0', 'M010-specific maximum current'),
    ('M010', 'sensor_thresholds', 'pressure_min', '3.2', 'M010-specific minimum pressure'),
    ('M010', 'sensor_thresholds', 'pressure_max', '5.8', 'M010-specific maximum pressure'),
    ('M010', 'sensor_thresholds', 'temperature_min', '62.0', 'M010-specific minimum temperature'),
    ('M010', 'sensor_thresholds', 'temperature_max', '88.0', 'M010-specific maximum temperature'),
    ('M010', 'sensor_thresholds', 'vibration_max', '2.5', 'M010-specific maximum vibration');
