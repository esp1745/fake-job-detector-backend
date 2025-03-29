from django.utils import timezone
from rest_framework import viewsets, status
from rest_framework.response import Response
import logging
from .models import JobPosting
from .serializers import JobPostingSerializer
from ml_model.predict import predict_job
from ml_model.web_scrapers import get_web_data

logger = logging.getLogger(__name__)

class JobPostingViewSet(viewsets.ModelViewSet):
    queryset = JobPosting.objects.all()
    serializer_class = JobPostingSerializer

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        instance = serializer.save()

        try:
            # Combine text data safely
            text_data = " ".join(filter(None, [
                instance.title,
                instance.company_profile,
                instance.description,
                instance.requirements
            ])).strip() or "No description provided."

            # Real-time web scraping for additional data
            web_data = get_web_data(
                company_name=instance.company_name,
                domain=instance.company_domain
            )

            # Construct metadata for prediction
            metadata = {
                'salary_missing': instance.salary_missing,
                'profile_length': instance.profile_length,
                'has_company_logo': instance.has_company_logo,
                'domain_age_days': web_data['domain_age_days'],
                'linkedin_exists': web_data['linkedin_exists'],
                'ssl_valid': web_data['ssl_valid'],
                'social_media_count': web_data['social_media_count']
            }

            # Perform prediction with the DL model
            prediction_result = predict_job(text_data, metadata)
            prediction = bool(prediction_result["prediction"])
            probability = prediction_result["probability"]

            # Generate human-readable explanation
            explanation = {
                'reasons': self._get_explanation_reasons(metadata),
                'metadata': metadata,
                'confidence': probability
            }

            # Update instance with predictions
            instance.prediction = prediction
            instance.is_fake = prediction
            instance.probability = probability
            instance.domain_age_days = metadata['domain_age_days']
            instance.linkedin_exists = metadata['linkedin_exists']
            instance.ssl_valid = metadata['ssl_valid']
            instance.social_media_count = metadata['social_media_count']
            instance.explanation = explanation
            instance.last_checked = timezone.now()
            instance.save()

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            instance.delete()
            return Response(
                {"error": "Job analysis failed"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        headers = self.get_success_headers(serializer.data)
        return Response(
            self.get_serializer(instance).data,
            status=status.HTTP_201_CREATED,
            headers=headers
        )

    def _get_explanation_reasons(self, metadata):
        """Generate clear reasons explaining prediction results"""
        reasons = []
        if metadata['salary_missing']:
            reasons.append('Salary information missing')
        if metadata['domain_age_days'] < 365:
            reasons.append('Domain is recently registered (less than 1 year old)')
        if not metadata['linkedin_exists']:
            reasons.append('No LinkedIn company profile found')
        if metadata['profile_length'] < 50:
            reasons.append('Company profile is short or missing')
        if not metadata['ssl_valid']:
            reasons.append('Company website lacks a valid SSL certificate')
        if metadata['social_media_count'] < 1:
            reasons.append('Company has low or no social media presence')

        return reasons