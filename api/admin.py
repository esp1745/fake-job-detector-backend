
from django.contrib import admin
from .models import JobPosting

@admin.register(JobPosting)
class JobPostingAdmin(admin.ModelAdmin):
    list_display = ('title', 'company_name', 'prediction', 'last_checked')
    readonly_fields = ('domain_age_days', 'linkedin_exists')