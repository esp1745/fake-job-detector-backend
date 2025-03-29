from django.db import models

class JobPosting(models.Model):
    # Core job post data
    title = models.CharField(max_length=200)
    description = models.TextField()
    company_name = models.CharField(max_length=100)
    company_domain = models.CharField(max_length=100)
    location = models.CharField(max_length=100)
    salary = models.CharField(max_length=50, blank=True)
    requirements = models.TextField(blank=True)
    company_profile = models.TextField(blank=True)
    has_company_logo = models.BooleanField(default=False)

    # Prediction results
    is_fake = models.BooleanField(default=False)
    prediction = models.BooleanField(default=False)
    probability = models.FloatField(null=True, blank=True)  # Prediction confidence
    explanation = models.JSONField(null=True, blank=True)  # Detailed reasons

    # Real-time web features (stored after checks for reference)
    domain_age_days = models.IntegerField(default=0)
    linkedin_exists = models.BooleanField(default=False)
    ssl_valid = models.BooleanField(default=False)
    social_media_count = models.IntegerField(default=0)

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    last_checked = models.DateTimeField(null=True, blank=True)

    # Calculated properties (not stored in DB)
    @property
    def salary_missing(self):
        return not bool(self.salary.strip()) if self.salary else True

    @property
    def profile_length(self):
        return len(self.company_profile.strip())

    def __str__(self):
        return f"{self.title} @ {self.company_name}"

    class Meta:
        indexes = [
            models.Index(fields=['company_domain']),
            models.Index(fields=['created_at']),
            models.Index(fields=['is_fake']),
        ]
