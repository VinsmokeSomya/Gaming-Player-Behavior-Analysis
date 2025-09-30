"""Configuration management for player retention analytics system."""

import os
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class DatabaseConfig(BaseModel):
    """Database configuration settings."""
    
    host: str = Field(default="localhost")
    port: int = Field(default=5432)
    name: str = Field(default="player_analytics")
    user: str = Field(default="analytics_user")
    password: str = Field(default="analytics_password")
    
    def __init__(self, **kwargs):
        # Load from environment variables
        data = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', '5432')),
            'name': os.getenv('DB_NAME', 'player_analytics'),
            'user': os.getenv('DB_USER', 'analytics_user'),
            'password': os.getenv('DB_PASSWORD', 'analytics_password'),
        }
        data.update(kwargs)
        super().__init__(**data)
    
    @property
    def connection_string(self) -> str:
        """Generate PostgreSQL connection string."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


class AppConfig(BaseModel):
    """Application configuration settings."""
    
    log_level: str = Field(default="INFO")
    environment: str = Field(default="development")
    model_retrain_threshold: float = Field(default=0.75)
    churn_prediction_threshold: float = Field(default=0.7)
    
    def __init__(self, **kwargs):
        # Load from environment variables
        data = {
            'log_level': os.getenv('LOG_LEVEL', 'INFO'),
            'environment': os.getenv('ENVIRONMENT', 'development'),
            'model_retrain_threshold': float(os.getenv('MODEL_RETRAIN_THRESHOLD', '0.75')),
            'churn_prediction_threshold': float(os.getenv('CHURN_PREDICTION_THRESHOLD', '0.7')),
        }
        data.update(kwargs)
        super().__init__(**data)


# Global configuration instances
db_config = DatabaseConfig()
app_config = AppConfig()