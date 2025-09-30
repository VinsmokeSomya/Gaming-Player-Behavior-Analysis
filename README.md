# Player Retention Analytics System

A comprehensive analytics system for mobile games that analyzes player behavior data, predicts churn risk, and provides actionable insights through visualizations.

## Features

- **Player Retention Analysis**: Calculate daily, weekly, and monthly retention rates
- **Churn Prediction**: Machine learning models to identify at-risk players
- **Interactive Visualizations**: Charts and reports for data-driven decisions
- **Drop-off Analysis**: Identify specific game levels where players quit
- **Automated Reporting**: Daily summaries and alerts for unusual patterns

## Project Structure

```
├── src/
│   ├── models/          # Data models and schemas
│   ├── services/        # Business logic and data processing
│   ├── analytics/       # Machine learning and analytics
│   ├── visualization/   # Charts and reporting
│   ├── config.py        # Configuration management
│   └── database.py      # Database connection utilities
├── sql/
│   └── init/           # Database schema and initialization
├── tests/              # Test suite
├── docker-compose.yml  # PostgreSQL database setup
├── requirements.txt    # Python dependencies
└── README.md
```

## Setup Instructions

### 1. Prerequisites

- Python 3.9+
- Docker and Docker Compose
- Git

### 2. Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd player-retention-analytics

# Create and activate virtual environment
python -m venv .venv

# On Windows
.venv\Scripts\activate

# On macOS/Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Database Setup

```bash
# Start PostgreSQL database
docker-compose up -d

# Wait for database to be ready (check logs)
docker-compose logs -f postgres

# Test database connection
python -m tests.test_database_connection
```

### 4. Environment Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your settings if needed
```

### 5. Verify Setup

```bash
# Run tests
pytest tests/

# Check database connection
python -c "from src.database import db_manager; print('✓ Connected' if db_manager.test_connection() else '✗ Failed')"
```

## Database Schema

The system uses PostgreSQL with the following main tables:

- **player_profiles**: Core player information and metrics
- **player_events**: Raw event data from game clients
- **player_sessions**: Aggregated session data
- **retention_metrics**: Calculated retention rates by cohort
- **churn_features**: Features for machine learning models

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_database_connection.py
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/
```

## Next Steps

After completing the setup, you can proceed with implementing the analytics features:

1. **Data Models**: Define Pydantic models for type safety
2. **ETL Pipeline**: Build data processing workflows
3. **ML Pipeline**: Implement churn prediction models
4. **Visualizations**: Create charts and dashboards
5. **API Layer**: Add REST endpoints for data access

## Troubleshooting

### Database Connection Issues

```bash
# Check if PostgreSQL is running
docker-compose ps

# View database logs
docker-compose logs postgres

# Reset database
docker-compose down -v
docker-compose up -d
```

### Python Environment Issues

```bash
# Recreate virtual environment
rm -rf .venv
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## License

This project is licensed under the MIT License.