# Installation Guide

## Quick Start (Demo Version)

The demo version works with built-in Python libraries and can be run immediately:

```bash
# Clone the repository
git clone https://github.com/VinsmokeSomya/Gaming-Player-Behavior-Analysis.git
cd Gaming-Player-Behavior-Analysis

# Run the demo
python3 demo_analysis.py
```

## Full Implementation Setup

For the complete analysis with scikit-learn and visualizations:

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt
```

If you encounter installation issues, install packages individually:

```bash
pip install pandas>=1.5.0
pip install numpy>=1.21.0
pip install scikit-learn>=1.2.0
pip install matplotlib>=3.5.0
pip install seaborn>=0.11.0
pip install sqlalchemy>=1.4.0
```

### Step 2: Run Full Analysis

```bash
# Run the complete analysis pipeline
python main_analysis.py
```

This will:
- Generate synthetic gaming data (10,000 players)
- Create SQL views for retention analysis
- Train machine learning models for churn prediction
- Generate comprehensive visualizations
- Display results and insights

### Step 3: Explore Results

After running the analysis, you'll find:

- **Database**: `data/gaming_data.db`
- **Trained Models**: `models/` directory
- **Visualizations**: `visualizations/` directory
- **Analysis Report**: Console output with key metrics

### Alternative: Jupyter Notebook

For interactive analysis:

```bash
# Install Jupyter if not already installed
pip install jupyter

# Start Jupyter notebook
jupyter notebook

# Open notebooks/Gaming_Player_Analysis_Demo.ipynb
```

## Troubleshooting

### Package Installation Issues

If you encounter installation problems:

1. **Update pip**: `pip install --upgrade pip`
2. **Use virtual environment**:
   ```bash
   python -m venv gaming_analysis
   source gaming_analysis/bin/activate  # On Windows: gaming_analysis\Scripts\activate
   pip install -r requirements.txt
   ```
3. **System packages**: Some systems may require additional packages:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install python3-dev python3-pip

   # CentOS/RHEL
   sudo yum install python3-devel python3-pip
   ```

### Memory Issues

For large datasets:
- Reduce the number of players in `src/data_generator.py`
- Use chunking for data processing
- Consider using a more powerful machine

### Visualization Issues

If matplotlib doesn't display plots:
- Install GUI backend: `pip install tkinter` (Linux) or ensure X11 forwarding
- Use `plt.savefig()` instead of `plt.show()` for headless environments
- Set matplotlib backend: `export MPLBACKEND=Agg`

## System Requirements

### Minimum Requirements
- **RAM**: 2GB available memory
- **Storage**: 1GB free space
- **CPU**: Any modern processor
- **OS**: Windows 10, macOS 10.14, or Linux

### Recommended Requirements
- **RAM**: 8GB+ for large datasets
- **Storage**: 5GB+ for extensive analysis
- **CPU**: Multi-core processor for faster model training
- **OS**: Latest stable versions

## Docker Alternative

For a containerized environment:

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "main_analysis.py"]
```

```bash
# Build and run
docker build -t gaming-analysis .
docker run -v $(pwd)/output:/app/output gaming-analysis
```

## Development Setup

For contributing or extending the project:

```bash
# Clone and setup development environment
git clone https://github.com/VinsmokeSomya/Gaming-Player-Behavior-Analysis.git
cd Gaming-Player-Behavior-Analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install development dependencies
pip install -r requirements.txt
pip install jupyter pytest black flake8

# Run tests
pytest tests/

# Format code
black src/
```

## Performance Optimization

For better performance:

1. **Use SSD storage** for database operations
2. **Increase RAM** for larger datasets
3. **Use multiprocessing** for parallel analysis
4. **Optimize SQL queries** with proper indexing
5. **Consider GPU acceleration** for large-scale ML training

## Support

If you encounter issues:

1. Check the [README.md](README.md) for project overview
2. Review [Issues](https://github.com/VinsmokeSomya/Gaming-Player-Behavior-Analysis/issues) on GitHub
3. Run the demo version first to verify basic functionality
4. Ensure all dependencies are properly installed

For additional help, please open an issue on GitHub with:
- Your operating system and Python version
- Full error message and traceback
- Steps to reproduce the problem