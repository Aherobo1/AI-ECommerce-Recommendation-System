"""
API Services Package

This package contains all API-related services including:
- Database services
- Cache services
- Performance monitoring
- External API integrations
"""

# Import existing services with fallback
try:
    from ..database import DatabaseService
except ImportError:
    DatabaseService = None

try:
    from ..cache_service import CacheService
except ImportError:
    CacheService = None

try:
    from ..performance_monitor import PerformanceMonitor
except ImportError:
    PerformanceMonitor = None

__all__ = [
    'DatabaseService',
    'CacheService',
    'PerformanceMonitor'
]
