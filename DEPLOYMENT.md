# Deployment Guide - Agentic Nudge

This guide covers multiple deployment options for the Agentic Nudge application.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Streamlit Cloud (Easiest)](#streamlit-cloud)
3. [Docker Deployment](#docker)
4. [Heroku](#heroku)
5. [Azure Web Apps](#azure)
6. [AWS (EC2 or ECS)](#aws)
7. [Environment Configuration](#environment-configuration)

---

## Prerequisites

- Git repository with your code
- OpenAI API key
- Python 3.11+ (for local testing)

---

## 1. Streamlit Cloud (Easiest) 🚀

**Perfect for demos and small-scale deployments**

### Steps:

1. **Push to GitHub:**
   ```bash
   git add .
   git commit -m "Prepare for deployment"
   git push origin main
   ```

2. **Deploy:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository and `ui.py` as the main file
   - Click "Advanced settings"

3. **Configure Secrets:**
   In the secrets section, add:
   ```toml
   OPENAI_API_KEY = "your-actual-api-key"
   ```

4. **Deploy!**
   - Click "Deploy"
   - Your app will be live at `https://your-app-name.streamlit.app`

**Pros:** Free, easy, automatic HTTPS, CI/CD built-in
**Cons:** Limited resources, public unless you have Teams plan

---

## 2. Docker Deployment 🐳

**Best for production and scalability**

### Local Testing:

```bash
# Build the image
docker build -t agentic-nudge .

# Run locally
docker run -p 8501:8501 \
  -e OPENAI_API_KEY="your-key" \
  agentic-nudge
```

Visit `http://localhost:8501`

### Deploy to Docker Hub:

```bash
# Tag and push
docker tag agentic-nudge your-dockerhub-username/agentic-nudge:latest
docker push your-dockerhub-username/agentic-nudge:latest
```

### Docker Compose (Recommended):

Create `docker-compose.yml`:
```yaml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "8501:8501"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    restart: unless-stopped
```

Run with:
```bash
docker-compose up -d
```

---

## 3. Heroku 🟣

**Good balance of ease and features**

### Steps:

```bash
# Install Heroku CLI if needed
# Windows: choco install heroku-cli
# Mac: brew tap heroku/brew && brew install heroku

# Login
heroku login

# Create app
heroku create your-app-name

# Set environment variable
heroku config:set OPENAI_API_KEY=your-key-here

# Deploy
git push heroku main

# Open app
heroku open
```

### Monitoring:
```bash
heroku logs --tail
```

**Cost:** Free tier available, then ~$7/month for hobby dyno

---

## 4. Azure Web Apps ☁️

**Best for enterprise deployments**

### Using Azure CLI:

```bash
# Install Azure CLI if needed
# https://docs.microsoft.com/en-us/cli/azure/install-azure-cli

# Login
az login

# Create resource group
az group create --name agentic-nudge-rg --location eastus

# Create app service plan
az appservice plan create \
  --name agentic-nudge-plan \
  --resource-group agentic-nudge-rg \
  --sku B1 \
  --is-linux

# Create web app
az webapp create \
  --resource-group agentic-nudge-rg \
  --plan agentic-nudge-plan \
  --name your-unique-app-name \
  --runtime "PYTHON:3.11"

# Configure environment variables
az webapp config appsettings set \
  --resource-group agentic-nudge-rg \
  --name your-unique-app-name \
  --settings OPENAI_API_KEY=your-key-here

# Deploy from local git
az webapp deployment source config-local-git \
  --resource-group agentic-nudge-rg \
  --name your-unique-app-name

# Add Azure remote and push
git remote add azure <url-from-previous-command>
git push azure main
```

### Using Container Registry:

```bash
# Create container registry
az acr create \
  --resource-group agentic-nudge-rg \
  --name yourregistry \
  --sku Basic

# Build and push
az acr build \
  --registry yourregistry \
  --image agentic-nudge:latest .

# Create web app from container
az webapp create \
  --resource-group agentic-nudge-rg \
  --plan agentic-nudge-plan \
  --name your-app-name \
  --deployment-container-image-name yourregistry.azurecr.io/agentic-nudge:latest
```

---

## 5. AWS Deployment 🟧

### Option A: EC2 Instance

```bash
# SSH into your EC2 instance
ssh -i your-key.pem ec2-user@your-instance-ip

# Install Docker
sudo yum update -y
sudo yum install docker -y
sudo service docker start
sudo usermod -a -G docker ec2-user

# Clone and run
git clone your-repo-url
cd agentic-nudge
docker build -t agentic-nudge .
docker run -d -p 80:8501 \
  -e OPENAI_API_KEY="your-key" \
  --restart unless-stopped \
  agentic-nudge
```

### Option B: ECS (Elastic Container Service)

```bash
# Push to ECR
aws ecr create-repository --repository-name agentic-nudge
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin your-account-id.dkr.ecr.us-east-1.amazonaws.com

docker tag agentic-nudge:latest \
  your-account-id.dkr.ecr.us-east-1.amazonaws.com/agentic-nudge:latest
docker push your-account-id.dkr.ecr.us-east-1.amazonaws.com/agentic-nudge:latest

# Create ECS task definition and service through AWS Console or CLI
```

---

## Environment Configuration

### Required Variables:

| Variable | Description | Where to Set |
|----------|-------------|--------------|
| `OPENAI_API_KEY` | Your OpenAI API key | **Required everywhere** |
| `PORT` | Port number | Auto-set by platforms |

### Optional Variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_MODEL` | gpt-4o-mini | LLM model to use |

### Security Best Practices:

1. **Never commit `.env` files**
2. **Use platform-specific secret management:**
   - Streamlit: Secrets in dashboard
   - Heroku: `heroku config:set`
   - Azure: Key Vault or App Settings
   - AWS: Secrets Manager or Parameter Store
   - Docker: Environment variables or secrets

3. **Rotate API keys regularly**

---

## Scaling Considerations

### Performance Tuning:

1. **For High Traffic:**
   - Use containerized deployment (Docker/Kubernetes)
   - Enable load balancing
   - Consider caching responses

2. **Cost Optimization:**
   - Monitor OpenAI API usage
   - Set up usage limits
   - Use cheaper models for non-critical operations

3. **Monitoring:**
   - Set up application logging
   - Monitor API rate limits
   - Track error rates

---

## Troubleshooting

### Common Issues:

**"OPENAI_API_KEY not found"**
- Ensure environment variable is set correctly
- Check spelling and casing
- Verify the key is valid

**Port binding errors**
- Ensure the port is not in use
- Check firewall settings
- Verify PORT environment variable

**Module import errors**
- Verify all dependencies in requirements.txt
- Check Python version (must be 3.11+)
- Rebuild Docker image if using containers

### Health Check Endpoints:

- Streamlit health: `/_stcore/health`
- Custom health checks can be added to `app.py`

---

## Next Steps

After deployment:

1. ✅ Test the application thoroughly
2. ✅ Set up monitoring and alerts
3. ✅ Configure custom domain (optional)
4. ✅ Enable HTTPS (automatic on most platforms)
5. ✅ Set up CI/CD pipeline (GitHub Actions included)
6. ✅ Monitor costs and usage

---

## Support

For deployment issues:
- Check logs: Platform-specific commands above
- Review [Streamlit docs](https://docs.streamlit.io/)
- Open an issue on GitHub

**Happy Deploying! 🚀**
