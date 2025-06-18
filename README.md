# 🚀 ManaKnight AI E-Commerce Recommendation System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)](https://flask.palletsprojects.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://tensorflow.org)
[![AI](https://img.shields.io/badge/AI-Powered-purple.svg)](https://github.com)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)](https://github.com)
[![Architecture](https://img.shields.io/badge/Architecture-Modular-blue.svg)](https://github.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**🎯 PROFESSIONAL AI-POWERED E-COMMERCE RECOMMENDATION SYSTEM**

A production-ready, modular AI e-commerce platform featuring advanced machine learning, computer vision, natural language processing, and intelligent product recommendations. Built with modern software engineering practices, comprehensive evaluation frameworks, and scalable architecture.

## ✨ Features - Production Ready

### 🎯 **Advanced AI/ML Capabilities**

- **🧠 Modular CNN Architecture**: Custom CNN classifier with base model interfaces and standardized prediction results
- **📝 OCR Text Extraction**: Advanced OCR processing with confidence scoring and error handling
- **🖼️ Computer Vision Pipeline**: Modular image preprocessing with configurable transformations and augmentation
- **🔍 Vector Database Integration**: Pinecone integration with intelligent fallback mechanisms
- **🤖 Intelligent Recommendations**: Context-aware product suggestions with comprehensive evaluation metrics

### 🏗️ **Professional Software Architecture**

- **⚡ Modular Design**: Clean separation of concerns with abstract base classes and interfaces
- **� Centralized Configuration**: Environment-based configuration management with validation
- **� Structured Logging**: Performance monitoring, API request tracking, and error logging with context
- **🧪 Comprehensive Testing**: Integration tests, unit tests, and automated evaluation frameworks
- **� Performance Benchmarking**: Automated benchmarking suite with historical tracking and comparison

### 🎨 **Production-Grade Features**

- **💎 Model Versioning**: Metadata tracking, model registry, and rollback capabilities
- **📱 Evaluation Framework**: Comprehensive metrics, confusion matrices, and performance analysis
- **🎪 Documentation**: Detailed architecture docs, API documentation, and evaluation notebooks
- **📋 Monitoring & Analytics**: Real-time performance metrics and system health monitoring

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Features](#-features)
- [Technology Stack](#-technology-stack)
- [Installation](#-installation)
- [API Documentation](#-api-documentation)
- [Project Structure](#-project-structure)
- [Usage Examples](#-usage-examples)
- [Development Status](#-development-status)
- [Module Details](#module-details)
- [Contributing](#-contributing)
- [License](#-license)

## 🛠 Technology Stack

### Backend

- **Framework**: Flask (Python web framework)
- **Vector Database**: Pinecone (for similarity search and recommendations)
- **OCR**: Tesseract (text extraction from images)
- **Machine Learning**: TensorFlow/Keras (custom CNN model development)
- **Web Scraping**: BeautifulSoup, Selenium
- **Image Processing**: OpenCV, PIL

### Frontend

- **HTML5/CSS3**: Responsive web interfaces
- **JavaScript**: Interactive user experience
- **Bootstrap**: UI components and styling

### Database & Storage

- **Vector Database**: Pinecone
- **File Storage**: Local filesystem (configurable for cloud storage)

### Development Tools

- **Package Management**: pip, requirements.txt
- **Version Control**: Git
- **Documentation**: Markdown, comprehensive docs in `/docs`
- **Containerization**: Docker, Docker Compose
- **Deployment**: Heroku (Procfile), Docker
- **Environment Management**: Virtual environment, setup scripts
- **Logging**: Centralized logging configuration in `/utils`

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git

### Step-by-Step Setup

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd ManaKnight-AI-ECommerce-Recommendation-System
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv

   # On Windows
   venv\Scripts\activate

   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the root directory:

   ```env
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_ENVIRONMENT=your_pinecone_environment
   FLASK_ENV=development
   FLASK_DEBUG=True
   ```

5. **Initialize the database and environment**

   ```bash
   python setup_environment.py
   ```

6. **Create sample data (optional)**

   ```bash
   python create_sample_data.py
   ```

7. **Run the application**
   ```bash
   python app.py
   ```

The application will be available at `http://localhost:5000`

## 📚 Documentation

### Additional Documentation Files

- **CONTRIBUTING.md** - Guidelines for contributing to the project
- **DEMO_SCRIPT.md** - Comprehensive demo script for presentations
- **IMPROVEMENTS_SUMMARY.md** - Summary of project improvements and enhancements
- **docs/model_architecture.md** - Detailed model architecture documentation
- **LICENSE** - MIT License terms and conditions

## 📚 API Documentation

### Base URL

```
http://localhost:5000
```

### Endpoints

#### 1. Product Recommendation Service

**POST** `/product-recommendation`

Process natural language queries and return product recommendations.

**Request:**

```json
{
  "query": "I need wireless headphones for gaming"
}
```

**Response:**

```json
{
  "products": [
    {
      "stock_code": "001",
      "description": "High-Quality Gaming Headphones",
      "unit_price": 89.99,
      "country": "USA",
      "similarity_score": 0.95
    }
  ],
  "response": "I found excellent gaming headphones that match your requirements...",
  "query_processed": "wireless headphones gaming"
}
```

#### 2. OCR-Based Query Processing

**POST** `/ocr-query`

Extract text from handwritten images and process as product queries.

**Request:**

- Form data with `image_data` file upload

**Response:**

```json
{
  "products": [...],
  "response": "Based on your handwritten query...",
  "extracted_text": "wireless mouse for office work",
  "confidence": 0.87
}
```

#### 3. Image-Based Product Detection

**POST** `/image-product-search`

Identify products from uploaded images using CNN model.

**Request:**

- Form data with `product_image` file upload

**Response:**

```json
{
  "products": [...],
  "response": "I identified this as a smartphone...",
  "detected_class": "smartphone",
  "confidence": 0.92
}
```

#### 4. Sample Response

**GET** `/sample_response`

Returns a sample HTML response showing the expected output format.

## 📁 Project Structure

```
ManaKnight-AI-ECommerce-Recommendation-System/
├── app.py                          # Main Flask application
├── config.py                       # Configuration management
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── setup_environment.py            # Environment setup script
├── create_sample_data.py           # Sample data creation
├── test_integration.py             # Integration testing
├── Dockerfile                      # Docker configuration
├── docker-compose.yml              # Docker Compose setup
├── Procfile                        # Heroku deployment
├── LICENSE                         # MIT License
├── CONTRIBUTING.md                 # Contribution guidelines
├── DEMO_SCRIPT.md                  # Demo script documentation
├── IMPROVEMENTS_SUMMARY.md         # Project improvements summary
├── .env                           # Environment variables (create this)
├── data/                          # Data storage
│   ├── dataset/                   # Dataset files
│   │   ├── dataset.csv           # Main e-commerce dataset
│   │   └── CNN_Model_Train_Data.csv # CNN training data
│   ├── ecommerce.db              # SQLite database
│   └── scraped_images/           # Web scraped product images
├── services/                      # Backend services
│   ├── __init__.py
│   ├── database.py               # Database operations
│   ├── recommendation.py         # Recommendation engine
│   ├── ocr_service.py           # OCR functionality
│   ├── cnn_model.py             # CNN model for image detection
│   ├── enhanced_cnn_model.py    # Enhanced CNN model service
│   ├── vector_db.py             # Vector database operations
│   ├── cache_service.py         # Caching layer service
│   ├── performance_monitor.py   # Performance monitoring
│   ├── data_cleaning.py         # Data cleaning utilities
│   ├── scraper.py               # Web scraping utilities
│   ├── api_services/            # API service modules
│   │   └── __init__.py
│   ├── data_processing/         # Data processing modules
│   │   ├── __init__.py
│   │   └── preprocessing_pipeline.py
│   └── ml_models/               # Machine learning models
│       ├── __init__.py
│       ├── base_model.py        # Base model interface
│       ├── cnn_classifier.py    # CNN classifier implementation
│       ├── evaluation_framework.py # Model evaluation framework
│       └── benchmarking_suite.py   # Performance benchmarking
├── models/                        # Trained models
│   ├── cnn_product_classifier.h5  # Main CNN model
│   ├── product_classifier.h5      # Alternative classifier
│   ├── product_vectors.pkl        # Vectorized product data
│   └── model_info.txt            # Model metadata
├── templates/                     # HTML templates
│   ├── index.html                # Main landing page
│   ├── sample_response.html      # Sample response template
│   ├── text_query.html          # Text query interface
│   ├── image_query.html         # Image query interface
│   └── product_upload.html      # Product upload interface
├── static/                       # Static files (CSS, JS, images)
│   ├── css/                     # Stylesheets
│   │   ├── style.css           # Main stylesheet
│   │   └── manaknight.css      # ManaKnight branding styles
│   ├── js/                     # JavaScript files
│   │   ├── main.js             # Main JavaScript
│   │   ├── text-query.js       # Text query functionality
│   │   ├── image-query.js      # Image query functionality
│   │   └── product-upload.js   # Product upload functionality
│   ├── images/                 # Static images
│   │   └── products/           # Product images
│   └── uploads/                # User uploaded files
├── notebooks/                    # Jupyter notebooks for development
│   ├── data_cleaning.ipynb      # Data cleaning notebook
│   ├── model_training.ipynb     # Model training notebook
│   ├── vector_database_setup.ipynb # Vector DB setup
│   └── comprehensive_model_evaluation.ipynb # Model evaluation
├── tests/                       # Unit tests
│   ├── test_api.py             # API endpoint tests
│   ├── test_services.py        # Service layer tests
│   ├── test_models.py          # Model functionality tests
│   └── test_complete_system.py # Complete system tests
├── utils/                      # Utility modules
│   └── logging_config.py       # Logging configuration
├── docs/                       # Documentation
│   └── model_architecture.md   # Model architecture documentation
├── logs/                       # Application logs
│   └── app.log                # Main application log
└── venv/                      # Virtual environment (local)
```

## 🎯 Usage Examples

### 1. Text Query Interface

```python
import requests

response = requests.post('http://localhost:5000/product-recommendation',
                        data={'query': 'affordable laptop for students'})
print(response.json())
```

### 2. Image Upload for OCR

```python
import requests

with open('handwritten_query.jpg', 'rb') as f:
    response = requests.post('http://localhost:5000/ocr-query',
                           files={'image_data': f})
print(response.json())
```

### 3. Product Image Detection

```python
import requests

with open('product_image.jpg', 'rb') as f:
    response = requests.post('http://localhost:5000/image-product-search',
                           files={'product_image': f})
print(response.json())
```

## 🎉 Development Status - 100% COMPLETE!

### ✅ **ALL MODULES COMPLETED (100%)**

#### **Module 1: Data Preparation & Backend Setup**

- [x] ✅ E-commerce dataset cleaning and preprocessing
- [x] ✅ Vector database creation (Pinecone + local fallback)
- [x] ✅ Similarity metrics implementation (cosine similarity)
- [x] ✅ Product recommendation service with natural language processing

#### **Module 2: OCR & Web Scraping**

- [x] ✅ OCR functionality implementation (Tesseract integration)
- [x] ✅ Web scraping for product images (automated collection)
- [x] ✅ OCR-based query processing with confidence scoring
- [x] ✅ Training dataset creation (CNN_Model_Train_Data.csv)

#### **Module 3: CNN Model Development**

- [x] ✅ CNN model training (10-category classification)
- [x] ✅ Image-based product detection with confidence scores
- [x] ✅ Model integration with vector database matching
- [x] ✅ Trained model file (models/cnn_product_classifier.h5)

#### **Module 4: Frontend Development & Integration**

- [x] ✅ Text query interface (beautiful, responsive design)
- [x] ✅ Image query interface (OCR processing)
- [x] ✅ Product image upload interface (CNN classification)
- [x] ✅ Professional UI with Mana Knight Digital branding

### � **PRODUCTION ENHANCEMENTS ADDED**

- [x] ✅ Real-time performance monitoring
- [x] ✅ Caching layer (Redis + memory fallback)
- [x] ✅ Comprehensive unit tests (API, services, models)
- [x] ✅ Error handling and security features
- [x] ✅ API documentation and health checks
- [x] ✅ System analytics and metrics collection

### 🏆 **READY FEATURES**

- [x] ✅ Live demonstration capabilities
- [x] ✅ Performance metrics dashboard
- [x] ✅ Professional documentation
- [x] ✅ Scalable architecture design
- [x] ✅ Production deployment ready

# Project Overview

This project is divided into four main modules, each focusing on a distinct aspect of the system's development. The modules are designed to work together seamlessly, culminating in a comprehensive solution for product recommendation, OCR-based query processing, and image-based product detection.

## Module 1: Data Preparation and Backend Setup

### Task 1: E-commerce Dataset Cleaning

- _Objective_: Ensure the dataset is clean and ready for analysis and vectorization.
- _Key Actions_: Remove duplicates, handle missing values, and standardize formats.

### Task 2: Vector Database Creation

- _Objective_: Set up a vector database using Pinecone to store product vectors.
- _Key Actions_: Define the database schema and integrate with Pinecone.

### Task 3: Similarity Metrics Selection

- _Objective_: Choose and justify the similarity metrics used to compare product vectors.
- _Key Actions_: Evaluate different metrics (e.g., cosine similarity, dot product) and select the best fit based on the dataset characteristics.

### Endpoint 1: Product Recommendation Service

- _Functionality_: Handle natural language queries to recommend products, including safeguards against bad queries and sensitive data exposure.
- _Input_: Customer's natural language query.
- _Output_: Product matches array and a natural language response within specified constraints.

## Module 2: OCR and Web Scraping

### Task 4: OCR Functionality Implementation

- _Objective_: Develop the capability to extract text from images using OCR technology.
- _Key Actions_: Integrate and configure an OCR tool (e.g., Tesseract).

### Task 5: Web Scraping for Product Images

- _Objective_: Scrape product images from e-commerce websites for training data `CNN_Model_Train_Data.csv`.
- _Key Actions_: Automate scraping, download images, and store them systematically and make sure you have enough data to train the CNN model.

### Endpoint 2: OCR-Based Query Processing

- _Functionality_: Extract and process handwritten queries using the same logic as Endpoint 1.
- _Input_: Image file with handwritten text.
- _Output_: Same output format as Endpoint 1, adapted for image inputs also return the extracted test from OCR.

## Module 3: CNN Model Development

### Task 6: CNN Model Training

- _Objective_: Develop a CNN model from scratch using only the `products` mentioned on `CNN_Model_Train_Data.csv` to identify products from images.
- _Key Actions_: Train the model using scraped images and clean data without using pre-trained models.

### Endpoint 3: Image-Based Product Detection

- _Functionality_: Use the CNN model to identify products from images and match them using the vector database.
- _Input_: Product image.
- _Output_: Product description and matching products in a format consistent with other endpoints. Also return the name of the `class` that you got from CNN model for the particular input image.

## Module 4: Frontend Development and Integration

### Frontend Page 1: Text Query Interface

- _Features_: Form to submit text queries, display natural language responses, and a product details table.

### Frontend Page 2: Image Query Interface

- _Features_: Allows users to upload images of handwritten queries and displays results similar to Page 1.

### Frontend Page 3: Product Image Upload Interface

- _Features_: Users can upload product images, and view the identified product description and related products in natural language and tabular format.

## Instructions for Presentation

### 1. Incremental Report Writing

Each module completion should be accompanied by a concise, to-the-point report that documents the process, decisions, and outcomes. These reports will be incremental, building upon each other as the bootcamp progresses.

#### Report Format Suggestion:

- _Title Page_: Include the module number and title, the names of the team members, and the submission date.
- _Introduction_: Briefly describe the objectives of the module and its importance to the overall project.
- _High-Level Flow_:
  - _Description_: Outline the main tasks and functionalities developed in the module.
  - _Diagrams_: Include flowcharts or diagrams that visually represent the architecture and data flow.
  - _Key Decisions_: Summarize crucial decisions made during the module, such as choice of technology, design patterns, and configurations.
- _Challenges and Solutions_:
  - Briefly discuss any challenges faced during the module and how they were addressed.
- _Conclusion_: Sum up the outcomes of the module and its readiness for integration with other modules.
- _References_: Cite any tools, libraries, or external resources that were used.

### 2. Video Documentation

Participants are required to create two sets of videos for each module, detailing both the functionality and the technical implementation. This will not only aid in a better understanding of the project but also serve as a reference for future projects.

#### Video Requirements:

- _Functional Demonstration Video_:
  - _Content_: Demonstrate the functionality of each endpoint and page developed in the module.
  - _Focus_: Show how the system responds to various inputs and scenarios. Explain the user interaction with the system.
  - _Duration_: Keep the video concise, preferably under 5 minutes.
- _Code Explanation Video_:
  - _Content_: Provide a high-level overview of the codebase for the module.
  - _Focus_: Explain the structure of the code, major classes, and functions. Highlight any significant patterns or algorithms used.
  - _Duration_: Limit the explanation to under 10 minutes.

### Submission Guidelines:

- _Timing_: Submit the videos along with the incremental report at the end of each module.
- _Format_: Ensure videos are in a common format (e.g., MP4) and quality is sufficient for clear viewing.
- _Hosting_: Upload videos to a platform accessible to all participants and reviewers (e.g., Google Drive, YouTube in unlisted mode). Or you can use loom, fluvid, vmaker etc alternatively.

## 🧪 Testing

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_api.py

# Run with coverage
python -m pytest --cov=services tests/

# Run integration tests
python test_integration.py

# Run complete system tests
python -m pytest tests/test_complete_system.py -v
```

### Test Structure

- `tests/test_api.py` - API endpoint tests
- `tests/test_services.py` - Service layer tests
- `tests/test_models.py` - Model functionality tests
- `tests/test_complete_system.py` - Complete system integration tests
- `test_integration.py` - Main integration testing script

## 🚀 Deployment

### Local Development

```bash
export FLASK_ENV=development
python app.py
```

### Production Deployment

```bash
# Using Gunicorn
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 app:app

# Using Docker
docker build -t manaknight-ecommerce-recommendation .
docker run -p 8000:8000 manaknight-ecommerce-recommendation

# Using Docker Compose
docker-compose up -d

# Deploy to Heroku
git push heroku main
```

### Environment Variables for Production

```env
FLASK_ENV=production
PINECONE_API_KEY=your_production_api_key
PINECONE_ENVIRONMENT=your_production_environment
SECRET_KEY=your_secret_key
DATABASE_URL=your_database_url
```

## 📊 Performance Metrics

### Expected Performance

- **Query Response Time**: < 500ms for text queries
- **OCR Processing**: < 2s for standard images
- **CNN Inference**: < 1s for product classification
- **Vector Search**: < 100ms for similarity matching

### Monitoring

- API response times
- Database query performance
- Model inference latency
- Error rates and exceptions

## 🔧 Configuration

### Configuration Management

The project uses a centralized configuration system via `config.py` that manages:

- Environment-specific settings (development, production, testing)
- Database configurations
- API keys and secrets
- Model parameters
- Logging levels

### Pinecone Setup

1. Create account at [Pinecone](https://www.pinecone.io/)
2. Create an index with appropriate dimensions
3. Add API key to environment variables

### OCR Configuration

```python
# Tesseract configuration
TESSERACT_CONFIG = {
    'lang': 'eng',
    'config': '--psm 6'
}
```

### CNN Model Configuration

```python
# Model parameters
MODEL_CONFIG = {
    'input_shape': (224, 224, 3),
    'num_classes': 50,
    'batch_size': 32,
    'epochs': 100
}
```

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

### Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`python -m pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Code Style

- Follow PEP 8 for Python code
- Use meaningful variable and function names
- Add docstrings for all functions and classes
- Keep functions small and focused

### Commit Messages

- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit first line to 72 characters
- Reference issues and pull requests when applicable

## 📝 Changelog

### Version 1.0.0 (In Development)

- Initial project setup
- Basic API structure
- Documentation framework
- Sample response templates

## 🐛 Known Issues

- OCR accuracy may vary with handwriting quality
- CNN model requires sufficient training data
- Vector database initialization may take time on first run

## 📞 Support

For support and questions:

- Create an issue in the repository
- Contact the development team
- Check the documentation for common solutions

## 🙏 Acknowledgments

- Pinecone for vector database services
- Tesseract OCR community
- Flask development team
- Open source contributors

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Team

- **Project Lead**: [Aherobo Ovie Victor]
- **Backend Developer**: [Aherobo Ovie Victor]
- **ML Engineer**: [Aherobo Ovie Victor]
- **Frontend Developer**: [Aherobo Ovie Victor]

---

**Note**: This project is part of a data science bootcamp and is designed for educational purposes. The system demonstrates various AI/ML techniques including NLP, computer vision, and recommendation systems.

## Instructions for Coding

### General Guidelines

- _Class-Based Implementation_: It is recommended to use class-based implementation for all backend services to ensure organized, reusable, and maintainable code.
- _Best Practices_:
  - _ACID Properties_: Ensure that database transactions are Atomic, Consistent, Isolated, and Durable to maintain data integrity and reliability.
  - _Modularity_: Build the codebase with clear modularity in mind. Separate different functionalities into distinct modules to enhance readability and maintainability.
- _Packaging_: Organize your code into packages that reflect the services they provide. This approach not only helps in maintaining the code but also simplifies the deployment and scaling process.
- Directories: All notebooks are organized in the `notebooks/` directory with proper naming conventions:
  - `data_cleaning.ipynb` - Data preprocessing and cleaning
  - `model_training.ipynb` - CNN model training and validation
  - `vector_database_setup.ipynb` - Vector database initialization
  - `comprehensive_model_evaluation.ipynb` - Complete model evaluation and metrics

### Tech Stack

- _Web Framework_: Use Flask for developing the backend. Flask provides flexibility and ease of use for setting up API services.
- _Vector Database_: Integrate Pinecone to manage and query vector data efficiently. Pinecone supports scalable vector searches which are crucial for the recommendation systems in this project.
