# ML Model Monitoring with Prometheus and Grafana

This directory contains the configuration files needed to set up monitoring for the Apple Quality Prediction ML model using Prometheus and Grafana.

## Setup Instructions

### Prerequisites

- Docker and Docker Compose installed
- The ML model application running

### Starting the Monitoring Stack

1. Navigate to the monitoring directory:

   ```
   cd monitoring
   ```

2. Start the monitoring stack using Docker Compose:

   ```
   docker-compose up -d
   ```

3. Access the monitoring tools:
   - Prometheus: http://localhost:9090
   - Grafana: http://localhost:3000

### Grafana Setup

1. Log in to Grafana with the default credentials:

   - Username: admin
   - Password: admin
   - You'll be prompted to change the password on first login

2. The Prometheus data source and ML Model dashboard should be automatically provisioned.

3. Navigate to Dashboards → ML Model Monitoring to view the pre-configured dashboard.

## Dashboard Features

The ML Model Monitoring dashboard includes:

- Total HTTP requests
- Average request duration
- Memory usage
- Total prediction requests
- Prediction quality metrics

## Customizing Dashboards

You can create additional dashboards or modify the existing one:

1. In Grafana, click the "+" icon in the sidebar
2. Select "Create Dashboard"
3. Add panels using metrics from Prometheus
4. Save your dashboard

## Troubleshooting

- If metrics aren't showing up, check that Prometheus can reach your application
- Verify the scrape configuration in `prometheus.yml`
- Check the Prometheus targets page at http://localhost:9090/targets

## Taking Screenshots

To capture a screenshot of your Grafana dashboard for submission:

1. Navigate to your dashboard
2. Adjust the time range to show meaningful data
3. Use your browser's screenshot functionality or press Print Screen
4. Save the image as `kstarid-grafana-dashboard.png` in the screenshots directory
