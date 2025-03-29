import whois
from datetime import datetime
import requests
import tldextract

def extract_main_domain(domain):
    """Extract main domain regardless of subdomains."""
    extracted = tldextract.extract(domain)
    return f"{extracted.domain}.{extracted.suffix}"

def check_ssl_valid(domain):
    """Check SSL validity of a domain."""
    try:
        response = requests.get(f'https://{domain}', timeout=5)
        return response.ok and response.url.startswith('https://')
    except requests.exceptions.SSLError:
        return False
    except requests.exceptions.RequestException:
        return False

def check_social_media_presence(company_name):
    """Check social media presence (simplified example)."""
    social_platforms = [
        f"https://twitter.com/{company_name}",
        f"https://facebook.com/{company_name}",
        f"https://instagram.com/{company_name}",
    ]
    count = 0
    for url in social_platforms:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                count += 1
        except requests.RequestException:
            continue
    return count

def get_web_data(company_name, domain):
    """Get domain age, LinkedIn presence, SSL validity, and social media presence."""
    results = {
        'domain_age_days': 0,
        'linkedin_exists': False,
        'ssl_valid': False,
        'social_media_count': 0
    }

    # Clean domain
    clean_domain = extract_main_domain(domain)

    # Domain age check
    try:
        domain_info = whois.whois(clean_domain)
        creation_date = domain_info.creation_date
        if isinstance(creation_date, list):
            creation_date = creation_date[0]
        results['domain_age_days'] = (datetime.now() - creation_date).days
    except Exception as e:
        print(f"Domain age error: {e}")

    # LinkedIn check
    try:
        formatted_name = company_name.strip().lower().replace(' ', '-')
        response = requests.get(
            f"https://linkedin.com/company/{formatted_name}",
            headers={'User-Agent': 'Mozilla/5.0'},
            timeout=5
        )
        results['linkedin_exists'] = response.status_code in [200, 999]
    except Exception as e:
        print(f"LinkedIn check error: {e}")

    # SSL validation
    results['ssl_valid'] = check_ssl_valid(clean_domain)

    # Social media presence
    results['social_media_count'] = check_social_media_presence(formatted_name)

    return results
