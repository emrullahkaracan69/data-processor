
# 📊 Data Processor - Advanced Data Preprocessing & Analysis Tool

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://data-processor.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A professional-grade web application for comprehensive data preprocessing, outlier detection, and exploratory data analysis built with Streamlit.

## 🚀 Live Demo
Try the app live: [data-processor.streamlit.app](https://data-processor.streamlit.app)

## ✨ Key Features

### 📁 Data Management
- **Multi-format Support**: Upload CSV and Excel files
- **Smart Data Loading**: Automatic encoding detection and error handling
- **Data Validation**: File size limits and format verification

### 🔍 Outlier Detection & Treatment
- **Multiple Methods**:
  - IQR (Interquartile Range) Method
  - Z-Score Analysis
  - Isolation Forest Algorithm
- **Treatment Options**:
  - Remove outliers
  - Cap outliers
  - Transform outliers
- **Visual Analysis**: Interactive plots showing outliers

### 📈 Data Visualization
- **Distribution Plots**: Histograms and box plots
- **Correlation Analysis**: Interactive heatmaps
- **Scatter Plots**: Relationship visualization
- **Statistical Summaries**: Comprehensive data profiling

### 🛠️ Data Processing
- **Missing Value Handling**: Multiple imputation strategies
- **Data Type Conversion**: Automatic and manual type casting
- **Feature Engineering**: Create new features from existing ones
- **Data Export**: Download processed data in CSV format

## 🏗️ Project Architecture
"""
data-processor/
│
├── app.py                 # Main Streamlit application
├── requirements.txt       # Project dependencies
├── README.md             # Project documentation
├── .gitignore           # Git ignore file
│
└── src/                 # Source code modules
    ├── __init__.py      # Package initializer
    ├── data_loader.py   # Data loading and validation
    ├── outlier_handler.py # Outlier detection algorithms
    └── utils.py         # Utility functions and helpers
"""
## 🔧 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

## 📖 Usage Guide

### 1. Upload Your Data
- Click on "Browse files" or drag and drop your CSV/Excel file
- Maximum file size: 200MB
- Supported formats: .csv, .xlsx, .xls

### 2. Explore Your Data
- View basic statistics and data types
- Check missing values
- See data distribution

### 3. Detect Outliers
- Select detection method (IQR, Z-score, or Isolation Forest)
- Configure parameters for each method
- View outliers in interactive visualizations

### 4. Process Your Data
- Choose treatment method for outliers
- Handle missing values
- Apply transformations

### 5. Download Results
- Export processed data as CSV
- Save visualizations as images

## 🛠️ Technologies Used

- **[Streamlit](https://streamlit.io/)** - Web application framework
- **[Pandas](https://pandas.pydata.org/)** - Data manipulation and analysis
- **[NumPy](https://numpy.org/)** - Numerical computing
- **[Plotly](https://plotly.com/)** - Interactive visualizations
- **[Scikit-learn](https://scikit-learn.org/)** - Machine learning algorithms
- **[Openpyxl](https://openpyxl.readthedocs.io/)** - Excel file handling

## 📊 Supported Outlier Detection Methods

### IQR Method
- Uses Q1 - 1.5×IQR and Q3 + 1.5×IQR boundaries
- Best for normally distributed data
- Customizable IQR multiplier

### Z-Score Method
- Identifies points beyond ±3 standard deviations
- Assumes normal distribution
- Adjustable threshold

### Isolation Forest
- Machine learning approach
- Works well with high-dimensional data
- Handles non-linear patterns

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (git checkout -b feature/AmazingFeature)
3. Commit your changes (git commit -m 'Add some AmazingFeature')
4. Push to the branch (git push origin feature/AmazingFeature)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

**Emrullah Karacan**
- GitHub: [@emrullahkaracan69](https://github.com/emrullahkaracan69)

## 🙏 Acknowledgments

- Streamlit team for the amazing framework
- Open source community for the excellent libraries
- Contributors and users for their valuable feedback

---

⭐ If you find this project useful, please consider giving it a star on GitHub!
