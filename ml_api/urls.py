from django.contrib import admin
from django.urls import path
from.views import classify_text, bulk_classify

urlpatterns = [
    path('admin/', admin.site.urls),
    path('classify/', classify_text, name='classify_text'),
    path('bulkclassify/', bulk_classify, name='classify_text')
]