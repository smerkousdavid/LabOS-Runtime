#!/bin/bash

generate_create_sql() {
    # Load environment variables from .env
    [ -f .env ] && export $(grep -v '^#' .env | xargs) || { echo "Error: .env file not found."; exit 1; }

    # Ensure required variables are set
    if [[ -z "$GROUP_KEY" || -z "$API_KEY" || -z "$MAIL" || -z "$PASSWORD" ]]; then
        echo "Error: GROUP_KEY, API_KEY, MAIL, or PASSWORD is not set in the .env file."
        exit 1
    fi

    # Hash the password using SHA-256
    PASSWORD_HASH=$(echo -n "$PASSWORD" | sha256sum | awk '{print $1}')

    # Define the output file path (resolve relative to this script's location)
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    CREATE_SQL_PATH="${SCRIPT_DIR}/create.sql"

    # Generate the create.sql content
    cat > "$CREATE_SQL_PATH" <<EOF
-- MySQL dump 10.13  Distrib 8.0.43, for macos15 (arm64)
--
-- Host: localhost    Database: shinobi_dev
-- ------------------------------------------------------
-- Server version	8.0.43

-- ==========================================================================================
-- Database and User Setup
-- ==========================================================================================
CREATE DATABASE IF NOT EXISTS shinobi_dev;

CREATE USER IF NOT EXISTS 'shinobi_dev_user'@'%' IDENTIFIED BY 'pass';
GRANT ALL PRIVILEGES ON shinobi_dev.* TO 'shinobi_dev_user'@'%';
FLUSH PRIVILEGES;

-- ==========================================================================================
-- Table structure for table \`API\`
-- ==========================================================================================
DROP TABLE IF EXISTS \`API\`;
CREATE TABLE \`API\` (
  \`ke\` varchar(50) DEFAULT NULL,
  \`uid\` varchar(50) DEFAULT NULL,
  \`ip\` varchar(255) DEFAULT NULL,
  \`code\` varchar(100) DEFAULT NULL,
  \`details\` text,
  \`time\` timestamp NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3;

-- Insert data into \`API\`
INSERT INTO \`API\` VALUES (
  '${GROUP_KEY}',
  '${UNIQUE_ID}',
  '0.0.0.0',
  '${API_KEY}',
  '{"treatAsSub":"0","permissionSet":"","monitorsRestricted":"0","auth_socket":"1","create_api_keys":"1","edit_user":"1","edit_permissions":"1","get_monitors":"1","edit_monitors":"1","control_monitors":"1","get_logs":"1","watch_stream":"1","watch_snapshot":"1","watch_videos":"1","delete_videos":"1","get_alarms":"1","edit_alarms":"1","monitorPermissions":{}}',
  NOW()
);

-- ==========================================================================================
-- Table structure for table \`Users\`
-- ==========================================================================================
DROP TABLE IF EXISTS \`Users\`;
CREATE TABLE \`Users\` (
  \`ke\` varchar(50) DEFAULT NULL,
  \`uid\` varchar(50) DEFAULT NULL,
  \`auth\` varchar(50) DEFAULT NULL,
  \`mail\` varchar(100) DEFAULT NULL,
  \`pass\` varchar(100) DEFAULT NULL,
  \`accountType\` int DEFAULT '0',
  \`details\` longtext,
  UNIQUE KEY \`users_mail_unique\` (\`mail\`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3;

-- Insert data into \`Users\`
INSERT INTO \`Users\` VALUES (
  '${GROUP_KEY}',
  '${UNIQUE_ID}',
  '${API_KEY}',
  '${MAIL}',
  '${PASSWORD_HASH}',
  0,
  '{"factorAuth":"0","size":"300000","days":"","event_days":"","log_days":"","max_camera":"","permissions":"all","edit_size":"1","edit_days":"1","edit_event_days":"1","edit_log_days":"1","use_aws_s3":"1","use_whcs":"1","use_sftp":"1","use_webdav":"1","use_discordbot":"1","use_ldap":"1","aws_use_global":"0","whcs_use_global":"0","b2_use_global":"0","webdav_use_global":"0","monitorOrder":{"undefinedundefined":{"x":"8","y":"0","height":"4","width":"4"},"${GROUP_KEY}cam1":{"ke":"${GROUP_KEY}","mid":"cam1","x":"4","y":"4","height":"4","width":"6"},"${GROUP_KEY}cam2":{"ke":"${GROUP_KEY}","mid":"cam2","x":"0","y":"0","height":"4","width":"5"}},"shinobihub":"0","shinobihub_key":"","factor_global_webhook":"1","factor_emailClient":"1","size_video_percent":"","size_timelapse_percent":"","size_filebin_percent":"","addStorage":{"/home/Shinobi/videos2/":{"name":"second","path":"/home/Shinobi/videos2/","limit":"","videoPercent":"","timelapsePercent":""}},"timelapseFrames_days":"","audio_note":"","audio_alert":"","audio_delay":"","event_mon_pop":"0","whcs_save":"0","whcs_endpoint":"","whcs_bucket":"","whcs_accessKeyId":"","whcs_secretAccessKey":"","whcs_region":"","whcs_log":"0","use_whcs_size_limit":"0","whcs_size_limit":"","whcs_max_days":"","whcs_dir":"","bb_b2_save":"0","bb_b2_bucket":"","bb_b2_accountId":"","bb_b2_applicationKey":"","bb_b2_log":"0","use_bb_b2_size_limit":"0","bb_b2_size_limit":"","bb_b2_max_days":"","bb_b2_dir":"","aws_s3_save":"0","aws_s3_bucket":"","aws_accessKeyId":"","aws_secretAccessKey":"","aws_region":"us-west-1","aws_storage_class":"STANDARD","aws_s3_log":"0","use_aws_s3_size_limit":"0","aws_s3_size_limit":"","aws_s3_max_days":"","aws_s3_dir":"","mnt_save":"0","mnt_path":"","mnt_log":"0","use_mnt_size_limit":"0","mnt_size_limit":"","mnt_max_days":"","mnt_dir":"","googd_save":"0","googd_credentials":"","googd_code":"","googd_log":"0","use_googd_size_limit":"0","googd_size_limit":"","googd_max_days":"","googd_dir":"","sftp_save":"0","sftp_host":"","sftp_port":"","sftp_username":"","sftp_password":"","sftp_privateKey":"","sftp_dir":"","clock_date_format":"","css":"","theme":"Ice-v3","global_webhook":"0","global_webhook_url":"","global_webhook_method":"GET","emailClient":"0","emailClient_host":"","emailClient_port":"","emailClient_secure":"0","emailClient_unauth":"0","emailClient_user":"","emailClient_pass":"","emailClient_sendTo":"","zwave":"0","zwave_host":"","zwave_user":"","zwave_pass":""}'
);

-- ==========================================================================================
-- End of File
-- ==========================================================================================
EOF

    echo "Generated $CREATE_SQL_PATH with GROUP_KEY, API_KEY, MAIL, and hashed PASSWORD."
}