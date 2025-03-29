from rest_framework import serializers
from .models import JobPosting

class JobPostingSerializer(serializers.ModelSerializer):
    class Meta:
        model = JobPosting
        fields = [
            'id',
            'title',
            'company_name',
            'company_domain',
            'location',
            'description',
            'salary',
            'requirements',
            'company_profile',
            'has_company_logo',
            'prediction',
            'probability',            # Include prediction confidence
            'explanation',            # Detailed explanation
            'domain_age_days',        # Real-time web feature
            'linkedin_exists',        # Real-time web feature
            'ssl_valid',              # Real-time web feature
            'social_media_count',     # Real-time web feature
            'created_at',
            'last_checked'
        ]
        read_only_fields = [
            'prediction', 
            'probability',
            'explanation', 
            'domain_age_days',
            'linkedin_exists',
            'ssl_valid',
            'social_media_count',
            'created_at',
            'last_checked'
        ]
