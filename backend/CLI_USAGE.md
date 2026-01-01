# University Assistant AI - CLI Usage Guide

## 🚀 Quick Start

### Launch Team Management CLI
```bash
# Start the comprehensive team management CLI
python team_cli.py

# Or run directly
python -m cli.main
```

### Original Chat Interface
```bash
# Start the original chat interface (now with enhanced auth)
python run.py
```

## 📋 CLI Command Structure

### Team Management Commands

#### View Team Structure
```bash
# List all teams and their roles
python team_cli.py admin teams list

# View specific team progress  
python team_cli.py admin teams progress --team=data-collection --detailed

# Generate team performance reports
python team_cli.py admin teams reports --format=table
```

#### Task Assignment & Management
```bash
# Assign task to team
python team_cli.py admin teams assign \
  --task="Optimize data collection pipeline" \
  --team=data-collection \
  --priority=high \
  --deadline=2024-01-15

# Show team progress
python team_cli.py admin teams progress --team=ai-model
```

### Operational Commands

#### Data Collection Operations
```bash
# Start data scraping
python team_cli.py ops scrape start --source=university --type=all

# Monitor scraping operations
python team_cli.py ops monitor metrics --component=scrapers --live

# Show system metrics dashboard
python team_cli.py ops monitor metrics
```

#### Data Processing Operations
```bash
# Run data processing pipeline
python team_cli.py workflows pipeline run --stage=processing

# Run full pipeline
python team_cli.py workflows pipeline run
```

### Workflow Management

#### Pipeline Operations
```bash
# Execute complete data pipeline
python team_cli.py workflows pipeline run

# Run specific pipeline stage
python team_cli.py workflows pipeline run --stage=collection
python team_cli.py workflows pipeline run --stage=processing  
python team_cli.py workflows pipeline run --stage=indexing
python team_cli.py workflows pipeline run --stage=validation
```

#### Reporting & Analytics
```bash
# Generate daily reports
python team_cli.py workflows report daily --team=all --format=standup

# Generate weekly reports  
python team_cli.py workflows report daily --team=data-collection --format=detailed

# Team-specific reports
python team_cli.py workflows report daily --team=ai-model
```

## 👥 Team-Specific Workflows

### Data Collection Team
**Team Lead:** Senior Data Engineer

```bash
# Execute data collection workflow
python -m cli.operations.scraper collect \
  --source=university \
  --pages=50 \
  --delay=2 \
  --validate

# Monitor scraping health
python -m cli.operations.scraper monitor --hours=48

# Schedule automated collection
python -m cli.operations.scraper schedule --time=02:00 --frequency=daily
```

### Data Processing Team  
**Team Lead:** ML Engineer

```bash
# Process collected data
python team_cli.py ops process clean --input=raw_data --output=cleaned

# Generate embeddings
python team_cli.py ops process embed --model=all-MiniLM-L6-v2

# Build vector index
python team_cli.py ops index build --rebuild=true
python team_cli.py ops index optimize --compression=true
```

### AI Model Team
**Team Lead:** AI/ML Engineer  

```bash
# Test model performance
python team_cli.py dev test load --model=groq-llama --concurrent=10

# Monitor model metrics
python team_cli.py ops monitor metrics --component=models

# Generate model quality report
python team_cli.py workflows report custom --metric=response-quality
```

### Quality Assurance Team
**Team Lead:** QA Engineer

```bash
# Run comprehensive test suite
python team_cli.py workflows qa test --suite=full

# Run integration tests
python team_cli.py dev test integration --coverage=80

# Security testing
python team_cli.py dev test security --scan=deep

# Validate system health
python team_cli.py workflows qa validate --threshold=95
```

### DevOps & Infrastructure Team
**Team Lead:** DevOps Engineer

```bash
# System status and health
python team_cli.py admin system status --detailed

# Monitor system alerts
python team_cli.py ops monitor alerts --critical-only

# Performance monitoring
python team_cli.py ops monitor performance --auto-scale

# Deployment workflows  
python team_cli.py workflows deploy production --rollback-plan
```

### Product & Project Management
**Team Lead:** Product Manager

```bash
# Assign tasks and track progress
python team_cli.py admin teams assign --task="Feature X development" --team=ai-model

# Generate comprehensive reports
python team_cli.py workflows report custom --kpi=all --period=month

# Track sprint progress
python team_cli.py admin teams progress --sprint=current

# Generate stakeholder reports
python team_cli.py admin teams reports --format=table
```

## 🔄 Daily Workflow Examples

### Morning Standup Automation
```bash
# Generate standup report for all teams
python team_cli.py workflows report daily --team=all --format=standup

# Check system health
python team_cli.py admin system status --detailed

# Review overnight pipeline results
python team_cli.py ops monitor metrics --component=pipeline
```

### Data Pipeline Execution
```bash
# Run morning data collection
python team_cli.py workflows pipeline run --stage=collection

# Process and index new data
python team_cli.py workflows pipeline run --stage=processing
python team_cli.py workflows pipeline run --stage=indexing

# Quality validation
python team_cli.py workflows qa test --priority=high
```

### Team Coordination
```bash
# Assign urgent tasks
python team_cli.py admin teams assign \
  --task="Fix production issue" \
  --team=devops-infrastructure \
  --priority=critical

# Monitor team progress
python team_cli.py admin teams progress --detailed

# Generate end-of-day reports
python team_cli.py workflows report daily --team=all
```

## 📊 Monitoring & Analytics

### Real-time Monitoring
```bash
# Live system metrics
python team_cli.py ops monitor metrics --live

# Component-specific monitoring
python team_cli.py ops monitor metrics --component=vector-db
python team_cli.py ops monitor metrics --component=ai-models
python team_cli.py ops monitor metrics --component=authentication
```

### Performance Analytics
```bash
# Team performance metrics
python team_cli.py admin teams progress --team=data-processing --detailed

# System performance trends
python team_cli.py ops monitor performance --trending --predictions

# Quality metrics
python team_cli.py workflows qa validate --threshold=95 --detailed
```

## 🛠️ Development Workflows

### Feature Development Process
```bash
# 1. Planning Phase
python team_cli.py admin teams assign --feature=new-feature --team=ai-model
python team_cli.py workflows qa test --baseline=current

# 2. Development Phase  
python team_cli.py dev debug trace --component=new-feature
python team_cli.py dev test unit --coverage=90 --watch

# 3. Testing Phase
python team_cli.py workflows qa test --feature=new-feature --full-suite
python team_cli.py dev test integration --new-feature

# 4. Deployment Phase
python team_cli.py workflows deploy staging --feature=new-feature
python team_cli.py workflows qa validate --staging
python team_cli.py workflows deploy production --approved
```

## 📈 Reporting & KPIs

### Team KPI Monitoring
Each team has specific KPIs tracked automatically:

**Data Collection Team:**
- Data freshness (< 24 hours)
- Scraping success rate (> 95%)
- Data quality score (> 90%)
- Coverage completeness (100% target pages)

**Data Processing Team:**
- Processing latency (< 5 minutes per document)  
- Embedding quality score (> 0.8)
- Vector index performance (< 100ms query)
- Data pipeline uptime (> 99%)

**AI Model Team:**
- Response accuracy (> 85%)
- Response time (< 3 seconds)
- Model availability (> 99.9%)
- User satisfaction rating (> 4.0/5.0)

**Quality Assurance Team:**
- Test coverage (> 90%)
- Bug escape rate (< 5%)
- Performance benchmarks (meet SLA)
- Security vulnerability count (0 critical)

**DevOps & Infrastructure Team:**
- System uptime (> 99.9%)
- Deployment success rate (> 95%)
- Mean time to recovery (< 30 minutes)
- Resource utilization efficiency (< 80%)

## 🔧 Configuration & Setup

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp .env.example .env
# Edit .env with your configuration

# Initialize database
python -c "from src.database import db_manager; db_manager.create_tables()"
```

### CLI Configuration
The CLI uses Rich for beautiful output and Click for command parsing. All team structures and workflows are defined in `cli/main.py` and can be customized based on your team's needs.

## 📞 Getting Help

### Command Help
```bash
# General help
python team_cli.py --help

# Specific command help
python team_cli.py admin teams --help
python team_cli.py ops scrape --help
python team_cli.py workflows pipeline --help
```

### Team Contact
Each team lead can be contacted for specific questions:
- **Data Collection:** Senior Data Engineer
- **Data Processing:** ML Engineer  
- **AI Model:** AI/ML Engineer
- **Quality Assurance:** QA Engineer
- **DevOps & Infrastructure:** DevOps Engineer
- **Product & Project Management:** Product Manager

This CLI framework provides a comprehensive management system for coordinating all aspects of the University Assistant AI project with clear team responsibilities and automated workflows.
