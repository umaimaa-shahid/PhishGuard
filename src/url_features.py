import ipaddress
import re
from urllib.parse import urlparse
import pandas as pd


def extract_url_features(raw_url, dataset=None):
    if not isinstance(raw_url, str) or not raw_url.strip():
        raise ValueError("Please enter a valid URL.")

    url = raw_url.strip()
    if not re.match(r"^https?://", url, flags=re.IGNORECASE):
        url = "https://" + url

    parsed = urlparse(url)
    host = parsed.hostname
    if not host:
        raise ValueError("Could not identify a hostname in this URL.")

    if dataset is not None and "URL" in dataset.columns:
        normalized = dataset["URL"].astype(str).str.strip()
        match = dataset.loc[normalized == raw_url.strip()]
        if not match.empty:
            return match.iloc[0:1].copy(), "Original dataset features"

    try:
        ipaddress.ip_address(host)
        is_ip = 1
    except ValueError:
        is_ip = 0

    host_parts = host.lower().split(".")
    if is_ip or len(host_parts) < 2:
        tld = ""
        subdomain_count = 0
        domain_body = host.lower()
    else:
        tld = host_parts[-1]
        subdomain_count = max(len(host_parts) - 2, 0)
        body_parts = host_parts[:-1]
        if body_parts and body_parts[0] == "www":
            body_parts = body_parts[1:]
        domain_body = ".".join(body_parts) if body_parts else host.lower()

    url_no_scheme = re.sub(r"^https?://", "", url, flags=re.IGNORECASE)
    url_len = max(len(url) - 1, 1)

    url_for_letters = re.sub(r"^www\.", "", url_no_scheme, flags=re.IGNORECASE)
    letters = max(sum(c.isalpha() for c in url_for_letters) - 1, 0)
    digits = sum(c.isdigit() for c in url_no_scheme)

    encoded = re.findall(r"%[0-9a-fA-F]{2}", url_no_scheme)
    encoded_count = len(encoded)

    raw_special = sum(
        not c.isalnum() and c not in {"=", "?", "&"}
        for c in url_no_scheme
    )
    other_special = max(raw_special - 1, 0)

    body_len = max(len(domain_body), 1)
    max_alpha = max((len(m) for m in re.findall(r"[A-Za-z]+", domain_body)), default=0)
    max_digit = max((len(m) for m in re.findall(r"[0-9]+", domain_body)), default=0)
    max_spec = max((len(m) for m in re.findall(r"[^A-Za-z0-9]+", domain_body)), default=0)
    continuation_rate = (max_alpha + max_digit + max_spec) / body_len

    features = {
        "URLLength": max(len(url) - 1, 0),
        "DomainLength": len(host),
        "IsDomainIP": is_ip,
        "TLD": tld,
        "CharContinuationRate": continuation_rate,
        "TLDLength": len(tld),
        "NoOfSubDomain": subdomain_count,
        "HasObfuscation": int(encoded_count > 0),
        "NoOfObfuscatedChar": encoded_count,
        "ObfuscationRatio": encoded_count / url_len,
        "NoOfLettersInURL": letters,
        "LetterRatioInURL": round(letters / url_len, 3),
        "NoOfDegitsInURL": digits,
        "DegitRatioInURL": round(digits / url_len, 3),
        "NoOfEqualsInURL": url_no_scheme.count("="),
        "NoOfQMarkInURL": url_no_scheme.count("?"),
        "NoOfAmpersandInURL": url_no_scheme.count("&"),
        "NoOfOtherSpecialCharsInURL": other_special,
        "SpacialCharRatioInURL": round(other_special / url_len, 3),
        "IsHTTPS": int(parsed.scheme.lower() == "https"),
    }

    
    return (
        pd.DataFrame([features]),
        "Calculated URL features"
    )
