"""
Setup script for Customer Satisfaction Prediction package
"""
from setuptools import setup, find_packages
import os
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements from requirements.txt
def read_requirements():
    """Read requirements from requirements.txt file"""
    requirements_path = this_directory / "requirements.txt"
    if requirements_path.exists():
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

# Package metadata
PACKAGE_NAME = "customer-satisfaction-prediction"
VERSION = "1.0.0"
DESCRIPTION = "Machine Learning system for predicting customer satisfaction from support ticket data"
AUTHOR = "ML Engineering Team"
AUTHOR_EMAIL = "ml-team@company.com"
URL = "https://github.com/company/customer-satisfaction-prediction"

# Requirements
INSTALL_REQUIRES = read_requirements()

# Development requirements
EXTRAS_REQUIRE = {
    'dev': [
        'pytest>=7.4.0',
        'pytest-cov>=4.1.0',
        'black>=23.0.0',
        'flake8>=6.0.0',
        'isort>=5.12.0',
        'mypy>=1.5.0',
        'pre-commit>=3.0.0',
    ],
    'docs': [
        'sphinx>=7.0.0',
        'sphinx-rtd-theme>=1.3.0',
        'myst-parser>=2.0.0',
    ],
    'deploy': [
        'gunicorn>=21.0.0',
        'docker>=6.0.0',
        'kubernetes>=27.0.0',
    ]
}

# Include all extra requirements in 'all'
EXTRAS_REQUIRE['all'] = [
    req for extra in EXTRAS_REQUIRE.values() for req in extra
]

# Classifiers for PyPI
CLASSIFIERS = [
    'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Developers',
    'Intended Audience :: Data Scientists',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Natural Language :: English',
    'Operating System :: OS Independent',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.11',
    'Programming Language :: Python :: 3.12',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Topic :: Scientific/Engineering :: Information Analysis',
    'Topic :: Software Development :: Libraries :: Python Modules',
    'Topic :: Office/Business :: Customer Relationship Management',
    'Framework :: Flask',
    'Environment :: Web Environment',
]

# Keywords for better discoverability
KEYWORDS = [
    'machine learning',
    'customer satisfaction',
    'prediction',
    'support tickets',
    'flask',
    'web application',
    'natural language processing',
    'scikit-learn',
    'xgboost',
    'data science',
    'customer analytics',
    'business intelligence'
]

# Package data to include
PACKAGE_DATA = {
    'customer_satisfaction_prediction': [
        'config/*.py',
        'app/templates/*.html',
        'app/static/css/*.css',
        'app/static/js/*.js',
        'data/raw/.gitkeep',
        'data/processed/.gitkeep',
        'data/models/.gitkeep',
        'reports/figures/*/.gitkeep',
    ]
}

# Data files to include
DATA_FILES = [
    ('config', ['config/config.py']),
    ('docs', ['README.md', 'requirements.txt']),
]

# Entry points for command-line tools
ENTRY_POINTS = {
    'console_scripts': [
        'customer-satisfaction-predict=main:main',
        'cs-predict=main:main',
        'cs-web=app.app:main',
        'cs-train=src.model_training:main',
        'cs-evaluate=src.model_evaluation:main',
        'cs-preprocess=src.data_preprocessing:main',
    ],
}

# Project URLs
PROJECT_URLS = {
    'Bug Reports': f'{URL}/issues',
    'Source': URL,
    'Documentation': f'{URL}/docs',
    'Funding': f'{URL}/sponsors',
    'Say Thanks!': f'{URL}/discussions',
}

# Setup configuration
setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    project_urls=PROJECT_URLS,
    
    # Package discovery
    packages=find_packages(exclude=['tests*', 'docs*', 'examples*']),
    package_data=PACKAGE_DATA,
    data_files=DATA_FILES,
    include_package_data=True,
    
    # Dependencies
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    python_requires='>=3.10',
    
    # Entry points
    entry_points=ENTRY_POINTS,
    
    # Metadata
    classifiers=CLASSIFIERS,
    keywords=KEYWORDS,
    license='MIT',
    platforms=['any'],
    zip_safe=False,
    
    # Additional metadata
    maintainer=AUTHOR,
    maintainer_email=AUTHOR_EMAIL,
    download_url=f'{URL}/archive/v{VERSION}.tar.gz',
    
    # Test suite
    test_suite='tests',
    tests_require=[
        'pytest>=7.4.0',
        'pytest-cov>=4.1.0',
        'pytest-mock>=3.11.0',
    ],
    
    # Command line interface
    scripts=[],
    
    # Package options
    options={
        'build_scripts': {
            'executable': '/usr/bin/python3',
        },
        'egg_info': {
            'tag_build': '',
            'tag_date': False,
        },
    },
)

# Post-installation message
def post_install():
    """Display post-installation message"""
    print("\n" + "="*60)
    print("✅ Customer Satisfaction Prediction installed successfully!")
    print("="*60)
    print("\n📋 Next steps:")
    print("1. Copy your dataset to data/raw/customer_support_tickets.csv")
    print("2. Run: cs-predict --mode full")
    print("3. Start web app: cs-web")
    print("4. Visit: http://localhost:5000")
    print("\n📚 Documentation: https://github.com/company/customer-satisfaction-prediction/docs")
    print("🐛 Issues: https://github.com/company/customer-satisfaction-prediction/issues")
    print("💬 Discussions: https://github.com/company/customer-satisfaction-prediction/discussions")
    print("\n🎉 Happy predicting!")
    print("="*60)

# Custom commands
from setuptools import Command

class PostInstallCommand(Command):
    """Post-installation for installation mode."""
    description = 'run post install actions'
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        post_install()

class TestCommand(Command):
    """Custom test command."""
    description = 'run tests'
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        import subprocess
        import sys
        
        # Run pytest
        result = subprocess.run([
            sys.executable, '-m', 'pytest', 
            'tests/', 
            '-v', 
            '--cov=src',
            '--cov-report=html',
            '--cov-report=term-missing'
        ])
        
        if result.returncode == 0:
            print("\n✅ All tests passed!")
        else:
            print("\n❌ Some tests failed!")
            sys.exit(1)

class LintCommand(Command):
    """Custom lint command."""
    description = 'run linting tools'
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        import subprocess
        import sys
        
        # Run linting tools
        tools = [
            ['black', '--check', 'src/', 'app/', 'config/'],
            ['flake8', 'src/', 'app/', 'config/'],
            ['isort', '--check-only', 'src/', 'app/', 'config/'],
            ['mypy', 'src/', 'app/', 'config/'],
        ]
        
        all_passed = True
        for tool in tools:
            print(f"\n🔍 Running {tool[0]}...")
            result = subprocess.run(tool, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ {tool[0]} passed")
            else:
                print(f"❌ {tool[0]} failed:")
                print(result.stdout)
                print(result.stderr)
                all_passed = False
        
        if all_passed:
            print("\n✅ All linting checks passed!")
        else:
            print("\n❌ Some linting checks failed!")
            sys.exit(1)

# Add custom commands to setup
setup.cmdclass = {
    'post_install': PostInstallCommand,
    'test': TestCommand,
    'lint': LintCommand,
}

# Validation function
def validate_setup():
    """Validate setup configuration"""
    import sys
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("❌ Error: Python 3.10+ required")
        sys.exit(1)
    
    # Check required files exist
    required_files = [
        'README.md',
        'requirements.txt',
        'main.py',
        'config/config.py',
        'src/__init__.py',
        'app/__init__.py',
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Error: Missing required files: {missing_files}")
        sys.exit(1)
    
    print("✅ Setup configuration validated successfully!")

# Run validation if executed directly
if __name__ == '__main__':
    validate_setup()
    
    # Display setup information
    print("\n" + "="*60)
    print("📦 CUSTOMER SATISFACTION PREDICTION SETUP")
    print("="*60)
    print(f"Package: {PACKAGE_NAME}")
    print(f"Version: {VERSION}")
    print(f"Description: {DESCRIPTION}")
    print(f"Author: {AUTHOR}")
    print(f"Python: {INSTALL_REQUIRES}")
    print(f"Dependencies: {len(INSTALL_REQUIRES)} packages")
    print(f"Entry Points: {len(ENTRY_POINTS['console_scripts'])} commands")
    print("="*60)
    
    # Run setup
    print("\n🚀 Installing package...")
    post_install()
