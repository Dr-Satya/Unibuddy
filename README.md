# University Assistant AI Chatbot

A production-grade, CLI-first AI chatbot with RAG (Retrieval-Augmented Generation), multi-model backends (Hugging Face & Groq), GDPR/HIPAA compliance modules, and threaded execution. Designed to be easily portable to a web widget later.

## 🚀 Features

- **Multi-Model Support**: Integrate with Hugging Face Transformers and Groq API
- **RAG Implementation**: Retrieval-Augmented Generation with vector embeddings
- **University Data Integration**: Automated scraping and processing of GD Goenka University data
- **HIPAA Compliance**: Built-in security, monitoring, and MFA modules
- **CLI Interface**: Rich command-line interface for prototyping and testing
- **Apache 2.0 License**: Free and open-source for commercial and academic use
- **Modular Architecture**: Easy to extend and customize

## 🏗️ Project Structure

```
uni-assistant/
├── README.md
├── LICENSE
├── .env.example
├── requirements.txt
├── pyproject.toml
├── run.py
├── src/
│   ├── __init__.py
│   ├── config.py           # Configuration management
│   ├── auth.py             # Authentication system
│   ├── database.py         # Database models and operations
│   ├── hipaa_mfa.py        # Multi-factor authentication
│   ├── hipaa_monitoring.py # Compliance monitoring
│   ├── hipaa_security.py   # Security utilities
│   ├── models.py           # AI model integrations
│   ├── rag.py              # RAG implementation
│   ├── scraper.py          # University data scraper
│   ├── services.py         # Business logic services
│   ├── main.py             # CLI interface
│   └── medical_example.json # Sample medical data
└── data/
    ├── raw/                # Raw scraped data
    ├── processed/          # Processed and chunked data
    └── vectordb/           # Vector database files
```

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd uni-assistant
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

## 🔧 Configuration

Edit the `.env` file with your configuration:

```env
# API Keys
HUGGINGFACE_API_TOKEN=your_huggingface_token_here
GROQ_API_KEY=your_groq_api_key_here

# University Data Settings
UNIVERSITY_URL=https://www.gdgoenkauniversity.com/admissions/fee-structure

# Model Configuration
DEFAULT_MODEL=mistral
MAX_TOKENS=2048
TEMPERATURE=0.7
```

## 🚀 Usage

### CLI Interface

Start the CLI chatbot:
```bash
python run.py
```

Or install and use as a package:
```bash
pip install -e .
uni-assistant
```

### Available Commands

- `chat` - Start interactive chat session
- `scrape` - Scrape university data
- `train` - Train/update RAG embeddings
- `status` - Check system status
- `help` - Show help information

## 🧠 AI Models

The assistant supports multiple AI models:

- **Mistral 7B** (Default) - Open-source, efficient
- **Llama 2** - High-quality responses
- **GPT-3.5/4** via Groq - Fast inference
- **Custom Fine-tuned Models** - University-specific training

## 🔍 RAG (Retrieval-Augmented Generation)

The RAG system:
1. **Scrapes** university data from official sources
2. **Processes** and chunks the content
3. **Embeds** text using sentence transformers
4. **Stores** vectors in FAISS/ChromaDB
5. **Retrieves** relevant context for queries
6. **Generates** informed responses

## 🔒 Security & Compliance

### HIPAA Compliance
- Data encryption at rest and in transit
- Audit logging and monitoring
- Multi-factor authentication
- Session management
- Data anonymization

### Security Features
- API key management
- Rate limiting
- Input validation
- SQL injection prevention
- XSS protection

## 📊 University Data Integration

Currently supports:
- **Fee Structure** from GD Goenka University
- **Admission Requirements**
- **Course Information**
- **Academic Calendar**

Planned integrations:
- Student portal data
- Faculty information
- Campus facilities
- Events and announcements

## 🛣️ Roadmap

### Phase 1: CLI Prototype ✅
- Basic CLI interface
- University data scraping
- RAG implementation
- Multi-model support

### Phase 2: Web Integration 🔄
- FastAPI web service
- REST API endpoints
- Web widget interface
- Real-time chat

### Phase 3: Advanced Features 📋
- Voice interface
- Multi-language support
- Advanced analytics
- Mobile app integration

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- [GD Goenka University](https://www.gdgoenkauniversity.com/)
- [Hugging Face Models](https://huggingface.co/models)
- [Groq API](https://console.groq.com/)

## 📧 Support

For support and questions, please open an issue on GitHub or contact the development team.

---

**Made with ❤️ for education and AI advancement**
