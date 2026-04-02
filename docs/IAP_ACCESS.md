# MIST API - Access Guide

## Production Access (Through IAP)

When IAP (Identity-Aware Proxy) is enabled on Cloud Run, you must access the API through the IAP-secured URL.

### Access Methods

#### 1. **Browser Access (for testing)**
- Navigate to: `https://mist-api-asjgzaju3a-uw.a.run.app`
- You'll be redirected to Google Sign-In
- After signing in, IAP adds the JWT token automatically
- You can then use the API with the same browser session

#### 2. **Programmatic Access (with OAuth)**

For scripts or applications, you need to obtain an IAP ID token:

```bash
# Using gcloud (if you're authenticated with Google)
gcloud auth print-identity-token --audiences=https://mist-api-asjgzaju3a-uw.a.run.app

# Use the token in your request
curl -X POST https://mist-api-asjgzaju3a-uw.a.run.app/query \
  -H "Authorization: Bearer $(gcloud auth print-identity-token --audiences=https://mist-api-asjgzaju3a-uw.a.run.app)" \
  -H "Content-Type: application/json" \
  -d '{"description": "engine misfire"}'
```

#### 3. **Service Account Access**

For service-to-service calls:

```python
import google.auth
from google.auth.transport.requests import Request
from google.oauth2 import id_token

# Get ID token for IAP
audience = "https://mist-api-asjgzaju3a-uw.a.run.app"
credentials, _ = google.auth.default()
auth_req = Request()
id_token = id_token.fetch_id_token(auth_req, audience)

# Use token in request
import requests
response = requests.post(
    "https://mist-api-asjgzaju3a-uw.a.run.app/query",
    headers={"Authorization": f"Bearer {id_token}"},
    json={"description": "engine misfire"}
)
```

### IAP Headers

When accessing through IAP, these headers are automatically added:
- `X-Goog-Iap-Jwt-Assertion`: The JWT token (verified by the API)
- `X-Goog-Authenticated-User-Email`: User's email
- `X-Goog-Authenticated-User-Id`: User's subject ID

## Development Mode

For local testing without IAP, set the `DEV_MODE` environment variable:

### Local Development

```bash
# Set DEV_MODE
export DEV_MODE=true

# Run the API locally
python -m src.api.server

# Test with just the email header
curl -X POST http://localhost:8000/query \
  -H "X-Goog-Authenticated-User-Email: ramsvandermeer@gmail.com" \
  -H "Content-Type: application/json" \
  -d '{"description": "engine misfire"}'
```

### Cloud Run (Temporary Dev Deployment)

**⚠️ Warning: Only use DEV_MODE for temporary testing. Never enable in production.**

```bash
# Deploy with DEV_MODE enabled
gcloud run deploy mist-api-dev \
  --image gcr.io/mist-487607/mist-api:latest \
  --set-env-vars="DEV_MODE=true" \
  --region=us-west1 \
  --project=mist-487607
```

## Security Notes

- **Production**: Always use IAP (DEV_MODE=false)
- **Development**: Use DEV_MODE only for local testing
- **JWT Verification**: In production, the API verifies the IAP JWT signature using Google's public keys
- **Header Spoofing**: In production, headers alone cannot bypass auth - the JWT must be valid

## Troubleshooting

### "Missing IAP JWT token" Error
- You're accessing the API directly without going through IAP
- Solution: Access through the IAP-secured URL or use an ID token

### "User not registered" Error
- Your email is not in the database
- Solution: Call `/auth/register` first to create an account

### "Account tier does not have API access" Error
- Your account is on the "blocked" tier (default for new users)
- Solution: Contact an admin to upgrade your tier
