import os
import ssl
import urllib3
import warnings

def configure_environment():
    """Configure environment for SSL and warnings"""
    # Disable SSL verification
    os.environ['PYTHONHTTPSVERIFY'] = '0'
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ['REQUESTS_CA_BUNDLE'] = ''
    
    # Remove problematic SSL vars
    for var in ['SSL_CERT_FILE', 'SSL_CERT_DIR', 'CERT_FILE']:
        if var in os.environ:
            del os.environ[var]
    
    # Disable warnings
    warnings.filterwarnings("ignore")
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    # Configure SSL context
    ssl._create_default_https_context = ssl._create_unverified_context