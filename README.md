# Cybersecurity-Focused Chatbot Project by : Youssef OUAAQA Fatima Zohra HAYOUN Lina TAACHA Mohamed BENDAHRASS Hicham ELGHALBZOURI

## Table of Contents
1. [Introduction](#introduction)
2. [Objectives](#objectives)
3. [Roles and Responsibilities](#roles-and-responsibilities)
   - [Front-end Developer (Angular)](#front-end-developer-angular)
   - [Backend / CI/CD / Security](#backend--cicd--security)
   - [LLMs Responsible / MLOps](#llms-responsible--mlops)
   - [Diffusion Model Responsible / MLOps](#diffusion-model-responsible--mlops)
   - [DevOps Responsible / Monitoring](#devops-responsible--monitoring)
4. [Technologies Used](#technologies-used)
5. [Setup Instructions](#setup-instructions)
6. [Threat Generation Model](#threat-generation-model)
   - [Overview](#overview)
   - [Dataset](#dataset)
   - [Model Training](#model-training)
   - [Training Process](#training-process)
   - [Results & Observations](#results--observations)
   - [Applications](#applications)
   - [Future Improvements](#future-improvements)
   - [Installation & Usage](#installation--usage)
   - [Acknowledgments](#acknowledgments)
7. [Microservices Architecture with Spring Boot and Flask](#microservices-architecture-with-spring-boot-and-flask)
   - [Overview of Microservices](#overview-of-microservices)
   - [Services](#services)
     - [Eureka Discovery Service](#eureka-discovery-service)
     - [Gateway Service](#gateway-service)
     - [Authentication Service using JWT](#authentication-service-using-jwt)
     - [Flask Application That Runs the Fine-tuning Model](#flask-application-that-runs-the-fine-tuning-model)
     - [Generative AI Service](#generative-ai-service)
     - [RAG Flask Service](#rag-flask-service)
8. [Front-End Hosting: Angular](#front-end-hosting-angular)
   - [Preparing the Application](#preparing-the-application)
9. [Installation and Configuration of Nginx](#installation-and-configuration-of-nginx)
   - [Step 1: Install Nginx](#step-1-install-nginx)
   - [Step 2: Configure Nginx](#step-2-configure-nginx)
   - [Step 3: Enable Configuration and Restart Nginx](#step-3-enable-configuration-and-restart-nginx)
10. [Hosting Spring Boot Services](#hosting-spring-boot-services)
    - [Step 1: Clone the Microservice Repository](#step-1-clone-the-microservice-repository)
    - [Step 2: Build the Project with Maven](#step-2-build-the-project-with-maven)
    - [Step 3: Deploy and Run the Service](#step-3-deploy-and-run-the-service)
11. [Hosting ML Services (Python)](#hosting-ml-services-python)
    - [Step 1: Install Prerequisites](#step-1-install-prerequisites)
    - [Step 2: Launch the Service](#step-2-launch-the-service)
    - [Result: Chatbot Hosted](#result-chatbot-hosted)
    - [Step 3: Deploy and Run the Service](#step-3-deploy-and-run-the-service)
12. [Front-End Development with Angular](#Front-End-with-Angular)
    - [Install Prerequisites](#step-1-install-prerequisites)
    - [Creating Components and Services](#Creating-Components-and-Services)
    - [Integrating with Backend Services](#Integrating-with-Backend-Services)
    - [User Interface Screenshots](#User-Interface-Screenshots)
   
13. [Conclusion](#conclusion)
   
## Introduction

This project aims to develop a cybersecurity-focused chatbot leveraging advanced machine learning technologies, including Language Understanding Models (LUMs) like GPT-4 and BERT, Diffusion Models for threat forecasting, and Retrieval-Augmented Generation (RAG) for enhanced responses. The system will adopt a microservices architecture to ensure scalability, maintainability, and ease of deployment. The project also integrates best practices in DevOps and MLOps for continuous integration, deployment, and monitoring.

## Objectives

The main objectives of the project are:

1. **Develop a Cybersecurity-Focused Chatbot**: Create an intelligent chatbot capable of understanding and answering cybersecurity-related queries, providing incident response, and assisting with security inquiries.
   
2. **Integrate Diffusion Models for Threat Forecasting**: Use Diffusion Models to simulate and predict the spread of threats or vulnerabilities within a network.

3. **Leverage Language Understanding Models (LUM)**: Utilize advanced models like GPT-4 and BERT to enable the chatbot to comprehend user input with high accuracy, specifically tailored for cybersecurity.

4. **Implement Advanced Retrieval-Augmented Generation (RAG)**: Improve chatbot responses by integrating external data sources such as threat intelligence and security advisories into the language models.

5. **Adopt Microservices Architecture**: Design the chatbot as a set of independent, deployable microservices to ensure scalability, ease of maintenance, and flexibility.

6. **Apply DevOps and MLOps Practices**: Ensure efficient development, testing, deployment, and monitoring by utilizing DevOps and MLOps methodologies, including CI/CD pipelines, containerization, model versioning, and infrastructure as code.


## Roles and Responsibilities

### Front-end Developer (Angular)
- Develop the chatbot's user interface using **Angular**.
- Ensure that the frontend is intuitive and responsive for security analysts to interact with the chatbot seamlessly.
- Integrate the frontend with the backend APIs for real-time communication.

### Backend / CI/CD / Security
- Implement backend services to handle business logic and data processing.
- Design and manage the **CI/CD pipelines** to automate the testing, building, and deployment of the chatbot's microservices.
- Ensure the security of the backend systems and integrate security measures, including encryption and secure API handling.

# LLMs Responsible / MLOps

As the LLMs Responsible, my role involves optimizing and fine-tuning pre-trained Language Understanding Models (LLMs) to ensure they accurately handle cybersecurity-related queries. In this project, we fine-tuned the **Llama-3-8B** model using a domain-specific cybersecurity dataset. This process is essential for adapting the model to our specific use case, enabling it to provide accurate, relevant, and actionable responses to cybersecurity questions.

---

## Fine-Tuning Process
![nlp_architecture_fine-tunning](https://github.com/user-attachments/assets/9d5968e0-f31c-488a-b299-365a2cb3571b)


### Step 1: Preparing the Dataset
To fine-tune the model, we curated a dataset tailored to cybersecurity. The dataset consists of question-answer pairs that cover a wide range of cybersecurity topics, such as threat detection, incident response, and network security. Additionally, synthetic data was generated using LLMs to enhance its diversity and coverage.

- **Dataset Source:** [Joussef/my_sec_data](https://huggingface.co/Joussef/my_sec_data) on Hugging Face  
- **Dataset Format:**
  ```json
  {
    "Question": "What is a common method used in Network Security to prevent unauthorized access?",
    "Answer": "A common method used in Network Security to prevent unauthorized access is through the use of firewalls. Firewalls act as a barrier between a trusted internal network and untrusted external networks, controlling incoming and outgoing network traffic based on predetermined security rules."
  }
- **Prompt Template:**
  ```text
  You are a cybersecurity assistant. You must only answer questions related to cybersecurity. If a question is unrelated to cybersecurity, politely refuse to answer.
  ### Instruction:
  {instruction}

  ### Input:
  {input}

  ### Response:
  {output}

### Step 2: Loading Pre-Trained Models

To begin the fine-tuning process, we loaded the **Llama-3-8B-Instruct** model using the **Unsloth framework**. This framework is designed to optimize the fine-tuning process for both speed and efficiency. The model was loaded with **4-bit quantization**, a technique that significantly reduces memory usage while maintaining high performance.

- **Model:** `unsloth/llama-3-8b-Instruct-bnb-4bit`  
- **Framework:** Unsloth (for faster fine-tuning)  
- **Quantization:** 4-bit (to reduce GPU memory requirements)  

### Step 3: Fine-Tuning the Model

The fine-tuning process involved training the model on a custom cybersecurity dataset. To ensure efficiency, we utilized **LoRA (Low-Rank Adaptation)**, which focuses on fine-tuning specific layers of the model while keeping the majority of the pre-trained weights frozen. This approach allows for parameter-efficient fine-tuning.

- **Training Framework:** `trl (SFTTrainer)`

#### LoRA Configuration:
- `r = 16` (rank of the low-rank matrices)
- `lora_alpha = 16`
- `target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`

#### Training Parameters:
- `batch_size = 2`
- `learning_rate = 1e-4`
- `max_steps = 1500`
- `gradient_accumulation_steps = 4`
- `warmup_steps = 5`

### Step 4: Evaluation and Testing

After fine-tuning, the model was thoroughly evaluated using a validation dataset to ensure it met performance requirements. The evaluation focused on the following key metrics:

- **Accuracy:** The model's ability to provide correct and relevant answers.
- **Relevance:** The model's adherence to cybersecurity-related queries.
- **Response Time:** The time taken to generate responses.

To verify the model's scope adherence, it was tested with both cybersecurity-related and unrelated questions. For example:

#### Example Tests:

- **Cybersecurity Query:**  
  _"What is a DoS attack?"_  
  **Response:**  
  _"A Denial of Service (DoS) attack is a type of cyber attack where an attacker attempts to make a computer or network resource unavailable by flooding it with traffic, causing it to become overwhelmed and unable to respond to legitimate requests."_  

- **Non-Cybersecurity Query:**  
  _"What is the capital of France?"_  
  **Response:**  
  _"I'm happy to help with cybersecurity-related questions, but I must politely refuse to answer questions that are unrelated to cybersecurity."_  

### Step 5: Saving and Deploying the Fine-Tuned Model

After successfully fine-tuning and evaluating the model, it was saved and prepared for deployment. This process ensures seamless integration into our cybersecurity chatbot.

- **Model and Tokenizer:** Saved locally for future use.
- **Hugging Face Hub Repository:** [`Joussef/llama_security_model`](https://huggingface.co/Joussef/llama_security_model)
- **Deployment Format:** Compatible with Hugging Face Transformers and GGUF, enabling efficient inference.

### Key Tools and Frameworks

- **Unsloth:** For faster and more efficient fine-tuning.
- **LoRA:** For parameter-efficient fine-tuning.
- **Hugging Face Transformers:** For model loading, training, and inference.
- **trl (SFTTrainer):** For supervised fine-tuning.
- **Hugging Face Hub:** For model storage and sharing.

### Results

The fine-tuned model demonstrates the following:

- **Domain-Specific Expertise:** Provides accurate and relevant responses to cybersecurity queries.
- **Scope Adherence:** Politely refuses to answer non-cybersecurity questions.
- **Efficiency:** Optimized for deployment in production environments.

### 2. Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation (RAG) is a robust architecture that combines a retrieval mechanism with generative capabilities, enhancing the ability of language models to produce accurate and contextually relevant responses. In this project, RAG is utilized to build a cybersecurity-focused chatbot that answers domain-specific queries by retrieving relevant context from a document database.
![image](https://github.com/user-attachments/assets/f5a86767-259a-428b-aa00-b5ca3d29c413)

#### Step 1: Embedding Function

The embedding function is a crucial component of RAG. It maps both documents and user queries into a shared vector space, facilitating efficient similarity-based searches. To ensure accurate retrieval, a pre-trained embedding model optimized for semantic understanding is used, capturing the nuanced meanings required for cybersecurity contexts.

- **Purpose:** Converts text data into numerical representations for efficient comparisons.
- **Key Advantage:** Enables fast and meaningful similarity searches between queries and document content.

#### Implementation

The embedding function is implemented as follows:

```python
import warnings
from langchain_community.embeddings import HuggingFaceEmbeddings

warnings.simplefilter(action='ignore', category=FutureWarning)

def get_embedding_function():
    # Use CPU for embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="OrdalieTech/Solon-embeddings-large-0.1",
        model_kwargs={'device': 'cpu'}  # Force CPU usage
    )
    return embeddings
```
#### Step 2: Populating the Database

Populating the document database is a critical step in setting up the retrieval mechanism for the chatbot. This involves preparing documents, processing them into manageable chunks, and embedding them into a vector database for efficient retrieval.

1. **Loading Documents:**
   - Documents are imported from a directory containing PDF files.
   - This enables seamless integration of real-world documents into the system, making the chatbot adaptable to dynamic content.

2. **Splitting Text into Chunks:**
   - To optimize retrieval, documents are divided into smaller, manageable chunks of text.
   - Overlapping chunks are created to ensure no loss of context and to maintain coherence during processing.

3. **Storing Embeddings in a Vector Database:**
   - The text chunks are embedded into a vector database, allowing retrieval based on semantic similarity to user queries.
   - A persistent vector database is utilized, ensuring embeddings are retained across sessions for consistent performance.

4. **Metadata Assignment:**
   - Unique identifiers are assigned to each text chunk based on its source document and position within the document.
   - This metadata ensures traceability and facilitates efficient retrieval of relevant information.

#### Step 3: Querying the Database

After populating the database, the system is ready to handle user queries by retrieving relevant context and generating precise responses. The workflow for querying the database is as follows:

1. **Retrieving Context:**
   - The system performs a similarity-based search to locate the most relevant text chunks from the database.
   - The top results are combined to form a comprehensive context that aligns with the user’s query.

2. **Generating a Response:**
   - The retrieved context is utilized to generate a tailored response to the user query.
   - A fine-tuned language model ensures the generated answers are both accurate and domain-specific, catering to cybersecurity-related queries.

3. **Providing Sources:**
   - In addition to the generated response, the system outputs the source documents referenced in the answer.
   - This feature enhances transparency, traceability, and trustworthiness by linking responses to their original sources.

### Key Features of RAG Implementation

1. **Efficient Retrieval**  
   - Utilizes a vector database (Chroma) to perform fast and accurate similarity-based searches for relevant text chunks.

2. **Domain-Specific Responses**  
   - Employs a fine-tuned language model (Llama-3) to generate precise, cybersecurity-focused answers tailored to user queries.

3. **Transparency**  
   - Enhances trust and traceability by providing the sources of retrieved information alongside the generated responses.


### Diffusion Model Responsible / MLOps
- Develop and deploy **Diffusion Models** that simulate the spread of threats and vulnerabilities across networks.
- Integrate diffusion models with the chatbot's backend to predict future security risks based on historical data.
- Apply MLOps best practices for model management, including continuous training, monitoring, and versioning.

### DevOps Responsible / Monitoring
- Implement **DevOps** practices for infrastructure automation, continuous integration, and continuous deployment.
- Use tools like **Docker**, **Kubernetes**, and **Terraform** to containerize and deploy microservices.
- Set up monitoring using **Prometheus**, **Grafana**, and **ELK Stack** to track system performance and ensure uptime.

## Technologies Used

- **Frontend**: Angular.
- **Backend**: SpringBoot, Python (Flask).
- **Language Models**: GPT-4, BERT
- **Diffusion Models**: Custom or pre-built models for threat diffusion forecasting
- **CI/CD**: Jenkins, GitLab CI, or GitHub Actions, gounicorn, nginx
- **Infrastructure as Code**: Terraform, Ansible
- **Model Management**: MLflow
- **Version Control**: Git, GitHub

## Setup Instructions

To set up the chatbot system locally, follow the steps below:

### 1. Clone the repository

```bash
git clone https://github.com/your-repository/cybersecurity-chatbot.git
cd cybersecurity-chatbot
```

# Threat Generation Model

## Overview

This project focuses on training a **threat generation model** using **GPT-2 (medium)** for Natural Language Processing (NLP). The model is designed to generate realistic cybersecurity threat descriptions based on a comprehensive dataset compiled from multiple sources, including **MITRE ATT&CK tactics, Common Vulnerabilities and Exposures (CVEs) from 1999-2024, and various cyber threat types** (e.g., SQL Injection, XSS, Ransomware, etc.).

## Dataset

The dataset consists of  **259,743 rows** , combining structured and unstructured threat intelligence data. It integrates:

* **MITRE ATT&CK tactics** : Each tactic represents a category of threats with descriptions.
* **CVEs (Common Vulnerabilities and Exposures)** : Includes CVE IDs with corresponding descriptions from 1999 to 2024.
* **General Cyber Threats** : Contains named cyber threats such as SQL Injection, Ransomware, XSS, along with their detailed descriptions.

### Example Data Structure

| Threat Name          | Description Threat                                                                                                  |
| -------------------- | ------------------------------------------------------------------------------------------------------------------- |
| SQL Injection Attack | SQL injection is a type of attack where an attacker injects malicious SQL code into a web application's database... |
| CVE-2008-4031        | Microsoft Office Word 2000 SP3, 2002 SP3, 2003 SP3, and 2007 Gold and SP1...                                        |
| Reconnaissance       | Adversaries may gather information about identities and roles within the victim organization...                     |
| Initial Access       | Adversaries may introduce computer accessories, networking hardware, or other computing devices...                  |

## Model Training

### Model: `gpt2-medium`

* The model was trained on the compiled dataset to learn cybersecurity threat patterns and generate new threat descriptions.
* **Training duration:** Each epoch took approximately 3 **hours** due to the dataset size.
* **Hardware:** The training process was conducted on a high-performance computing setup to handle the extensive data.

## Training Process

1. **Data Preprocessing:**
   * Cleaned and formatted data to ensure consistency.
   * Tokenized text and prepared input sequences for GPT-2.
2. **Fine-Tuning GPT-2:**
   * Used **transfer learning** on `gpt2-medium` with a  **causal language modeling objective** .
   * Employed **gradient accumulation** to manage memory usage.
3. **Evaluation:**
   * Evaluated generated text based on coherence, relevance, and accuracy.
   * Compared generated outputs with real-world threat descriptions.

## Results & Observations

* The model successfully learned patterns in cybersecurity threat descriptions and generated meaningful threat reports.
* Some challenges included:
  * Handling rare or complex threat descriptions.
  * Avoiding redundant threat explanations.
  * Ensuring descriptions remained realistic and did not generate misleading information.
* The generated threats closely resembled real-world cybersecurity threats from the dataset.

## Applications

This model can be useful for:

* **Cybersecurity Research** : Generating synthetic cybersecurity threats for testing and training purposes.
* **Threat Intelligence Automation** : Assisting in generating new threat descriptions for security analysts.
* **Adversarial Simulation** : Developing realistic threat scenarios for cybersecurity training exercises.

## Future Improvements

* Experimenting with **larger transformer models** (e.g., GPT-3, Llama) for improved threat description accuracy.
* Integrating **adversarial training** to detect and mitigate model-generated hallucinations.
* Enhancing **dataset diversity** by incorporating more structured threat intelligence sources.

## Installation & Usage

### Prerequisites

* Python 3.8+
* PyTorch
* Hugging Face Transformers
* Pandas, NumPy, and other dependencies

### Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/linataataa/Ml_chatbot.git
   cd ML_chatbot
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Run model inference:

   ```bash
   python generative_AI/app_gen_ai.py --input "SQL Injection Attack"
   ```

## Acknowledgments

* **MITRE ATT&CK** for providing structured threat intelligence.
* **CVE Database** for cybersecurity vulnerability data.
* **Hugging Face** for pre-trained NLP models and libraries.



# Microservices Architecture with Spring Boot and Flask  

This project demonstrates a **microservices architecture** using Spring Boot for the core backend services and Flask for serving a fine-tuning machine learning model. Each service in the architecture is designed to be independently deployable, scalable, and maintainable.  

## Overview of Microservices  

Microservices architecture in Spring Boot is an approach to building an application as a collection of small, independently deployable services. Each service focuses on a specific business capability and communicates with others using lightweight protocols, typically HTTP. These services are often hosted in containers for scalability.  

### Key Features:  
- Independent services for specific functionalities.  
- Centralized service discovery using **Eureka**.  
- Gateway routing for efficient communication between clients and services.  
- JWT-based authentication for secure user management.
- Generative AI Service
- RAG service  

---

## Services  

### 1. Eureka Discovery Service  

**Eureka** is a service registry provided by Netflix, integrated with Spring Cloud, that maintains a registry of all available services. It eliminates the need to hardcode service endpoints by allowing services to dynamically register and discover one another.  

#### Features:  
- **Service Registration:** Services automatically register themselves with the Eureka server when they start.  
- **Service Discovery:** Services query Eureka to dynamically locate other services.  

---

### 2. Gateway Service  

The **Gateway Service** serves as the single entry point for all client requests to the microservices. It routes incoming requests to the appropriate microservice and manages aspects such as security and monitoring.  

#### Features:  
- **Routing:** Forwards client requests based on route configuration.  
- **Security:** Integrates with authentication mechanisms like JWT.  

#### Example Configuration in `application.yml`:  
```yaml  
routes:  
  - id: users  
    uri: http://localhost:8090  
    predicates:  
      - Path=/user/**  
  - id: ftmodel  
    uri: http://localhost:5000  
    predicates:  
      - Path=/ft/**  
  - id: gar_gen  
    uri: http://localhost:5002  
    predicates:  
      - Path=/rag/**  
  - id: ai_gen  
    uri: http://localhost:5001  
    predicates:  
      - Path=/ai/**  
```
### 3. Authentication Service using JWT

The **Authentication Service** is responsible for verifying user credentials and generating **JWT (JSON Web Token)** tokens. JWT is a compact and self-contained token used for securely transmitting information between parties.

### Key Steps in JWT-Based Authentication

#### 1. **User Login**  
The user sends their credentials (e.g., `username` and `password`) to the **Authentication Service** via a login endpoint.

#### 2. **Token Generation**  
If the credentials are valid, the service generates a JWT token containing claims such as:  
- User roles (e.g., `ADMIN`, `USER`)  
- Expiration time  
- User-specific identifiers (e.g., `userID`)  

The token is signed using a secure key to prevent tampering.

#### 3. **Token Usage**  
The client includes the JWT token in the headers of their requests to other services. For example:  

### Flask Application That Runs the Fine-tuning Model

The Flask application serves as a lightweight API server that runs a machine learning model, making predictions based on input data. This application is part of a microservices architecture, focusing on serving model predictions via an API endpoint.

#### Service Components:

1. **Model Initialization:**
   - The model is loaded using the `llama_cpp` library, which interfaces with a pre-trained machine learning model, `unsloth-llama-3-q4_k_m`. The model is initialized using the `Llama.from_pretrained()` method. Inference is then performed by calling the `model.create_chat_completion()` function to generate responses based on input prompts.

2. **Eureka Registration:**
   - The Flask app is registered with the **Eureka** server as a microservice. During registration, the app sends service details such as the instance ID, host name, port, and status. This enables other services in the architecture to dynamically discover the Flask app through the **Eureka Discovery Service**.

3. **API Endpoint for Generating Responses:**
   - The Flask application exposes a POST endpoint at `/generate` that accepts a `prompt` as input and returns a response generated by the model. 
   - **Request Example:**
     ```json
     {
       "prompt": "---------------------"
     }
     ```
   - **Response Example:**
     ```json
     {
       "response": "-------------------------"
     }
     ```


### 4. Generative AI Service

The Generative AI Service is a Flask-based microservice designed to detect and generate insights on cybersecurity threats using an advanced AI model. It forms part of a microservices architecture and is registered with Eureka for seamless integration with other services.

## Service Features

### Model Initialization
- The service uses a pre-trained GPT-2 model (`Mohammedbendahrass/threat-detection-gpt2`) fine-tuned for threat detection tasks.
- The model and tokenizer are initialized using the `transformers` library and are optimized for execution on the CPU.

### Eureka Registration
- The Flask app is dynamically registered with the Eureka Discovery Service using an instance ID and service details such as port and hostname.
- Registration ensures other microservices can discover and communicate with this service without hardcoded endpoints.

### API Endpoint for Threat Detection
- **Endpoint**: `/ai/detection`
- **Method**: `POST`
- **Input**: Accepts a JSON payload with a `prompt` containing a cybersecurity threat description.
  - Example:
    ```json
    {
      "prompt": "THREAT_NAME: Ransomware attack\nTHREAT_DESCRIPTION:"
    }
    ```
- **Output**: Returns a generated response detailing the threat or providing additional insights.
  - Example:
    ```json
    {
      "response": "THREAT_DESCRIPTION: This ransomware attack encrypts user files and demands a ransom payment in cryptocurrency to decrypt them.",
      "status": "success"
    }
    ```





### 5. RAG Flask Service

## Overview
The RAG Flask Service integrates retrieval-augmented generation (RAG) techniques with a language model to provide domain-specific, context-aware responses. This implementation focuses on the cybersecurity domain, enhancing response accuracy by leveraging external data sources for context retrieval.

---

## Features and Capabilities

### Service Purpose
- Designed to answer cybersecurity-related questions using a hybrid approach of retrieval and generation.
- Combines a pre-trained large language model (LLM) with a Chroma vector database for relevant context retrieval.

### Key Functionalities

1. **Context Retrieval**:
   - The service uses a Chroma vector store to retrieve relevant documents based on the query.
   - Documents are retrieved using similarity search with a custom embedding function.

2. **Contextual Response Generation**:
   - A pre-defined prompt template ensures the LLM focuses on cybersecurity-specific answers.
   - The retrieved documents serve as context for the LLM to generate informed responses.

3. **Source Attribution**:
   - The response includes references to the source documents, enhancing transparency and reliability.

### Eureka Registration
- Dynamically registers with the Eureka Discovery Service.
- Facilitates service discovery in a microservices architecture, ensuring seamless communication between components.

---

## Implementation Details

### Input

- **Endpoint**: `/rag/generate`
- **Method**: `POST`
- **Payload**:
  ```json
  {
    "query_text": "What are the primary risks of ransomware attacks?"
  }


### Output
 ```json
{
  "response": "Ransomware attacks encrypt user data and demand payment for decryption. Primary risks include data loss, operational downtime, and financial extortion.",
  "sources": ["doc_123", "doc_456"]
}
 ```
# Front-End Hosting: Angular

## 1. **Preparing the Application**

To prepare our Angular application for deployment, we need to generate the optimized files for production. We can do this by running the following command:

```bash
ng build --prod
```

---

# Installation and Configuration of Nginx

Nginx is a high-performance, open-source web server that is designed to handle heavy loads while maintaining speed and efficiency. It's widely used for modern applications due to its reliability, flexibility, and scalability.

For hosting our Angular application on AWS EC2 t3.large instance, follow these steps:

### **Step 1: Install Nginx**

1. Update the system's package list:
   ```bash
   sudo apt update
   ```

2. Install Nginx:
   ```bash
   sudo apt install nginx
   ```

---

### **Step 2: Configure Nginx**

We configure Nginx for our Angular project by creating a new configuration file:

1. Open the configuration file at `/etc/nginx/sites-available/angular-project`:

   ```bash
   sudo nano /etc/nginx/sites-available/angular-project
   ```

2. Add the following configuration:

   ```nginx
   server {
       listen 80;
       server_name 13.48.34.100; 

       root /var/www/angular_chatbot/front-chatbot/dist/front-chatbot/browser ; # Path to Angular app files
       index index.html;

       location / {
           try_files $uri $uri/ /index.html; # Serve index.html for Angular routes
       }

       # Optional: Enable gzip compression for improved performance
       gzip on;
       gzip_types text/plain application/javascript application/x-javascript text/javascript text/css application/json;
       gzip_min_length 1000;
   }
   ```

---

### **Step 3: Enable Configuration and Restart Nginx**

1. Create a symbolic link from the configuration to the **sites-enabled** folder:

   ```bash
   sudo ln -s /etc/nginx/sites-available/angular-project /etc/nginx/sites-enabled/
   ```

2. Restart Nginx:

   ```bash
   sudo systemctl restart nginx
   ```

---

# Hosting Spring Boot Services

To host our microservices developed with Spring Boot, we followed a simple and efficient process. Below are the detailed steps:

### **Step 1: Clone the Microservice Repository**

Each microservice is managed in a separate Git repository. Start by cloning the repository containing the microservice's source code:

```bash
git clone https://github.com/linataataa/springboot_services.git
cd springboot_services
```

### **Step 2: Build the Project with Maven**

Once the repository is cloned, use Maven to compile and package the application, which will generate an executable `.jar` file.

Run the following command:

```bash
mvn clean package
```

After the build completes, a `.jar` file is generated in the `target/` directory.

---

### **Step 3: Deploy and Run the Service**

To run the Spring Boot microservice, use the following command:

```bash
nohup java -jar /target/notre-application.jar > /var/log/spring-services.log 2>&1 &
```

The service will now be running in the background.

![Spring Boot Microservice](https://github.com/user-attachments/assets/e893929f-f20b-413b-b65e-826a32bf2f04)

---

# Hosting ML Services (Python)

Our Machine Learning services are hosted using **Gunicorn**. The following steps outline how to deploy and execute the Python-based services, similar to the Spring Boot process.

### **Step 1: Install Prerequisites**

Make sure the following tools are installed:

- python3  
- pip3  
- gunicorn  

### **Step 2: Launch the Service**

To launch the Python service with Gunicorn, run:

```bash
nohup gunicorn --bind 0.0.0.0:5000 --workers 5 app:app > /var/log/python-services.log 2>&1 &
```

This command will start the service and handle multiple requests with 5 workers, each running independently.

![ML Service](https://github.com/user-attachments/assets/a474833d-d9bf-4aab-9884-ed91e154ca17)

---

### **Result: Chatbot Hosted**

The chatbot is now hosted successfully:

![Hosted Chatbot](https://github.com/user-attachments/assets/3a810928-d4b0-4e7f-81d0-1ad49c515aea)

---

# Front-End Development with Angular

## Introduction
In this project, the front-end of the cybersecurity-focused chatbot is developed using **Angular**, a powerful and widely-used framework for building dynamic web applications. Angular provides a robust structure for creating reusable components, services, and modules, making it an ideal choice for developing complex user interfaces like the one required for this chatbot.

---

## Angular Project Structure
The Angular project is structured to ensure modularity, scalability, and maintainability. Below is an overview of the key components and services used in the front-end development:

### Key Components

1. **App Component**:
   - **`app.component.ts`**: The main component that serves as the entry point for the application. It handles the initial rendering and routing.
   - **`app.component.html`**: The template for the main component, defining the structure of the application's UI.
   - **`app.component.css`**: Styles specific to the main component.

2. **Chat Component**:
   - **`chat.component.ts`**: Manages the chat interface, including sending and receiving messages.
   - **`chat.component.html`**: Defines the layout of the chat window, including input fields and message display areas.
   - **`chat.component.css`**: Styles for the chat interface.

3. **Home Component**:
   - **`home.component.ts`**: Handles the logic for the home page, including navigation and initial user interactions.
   - **`home.component.html`**: The template for the home page, providing a welcoming interface for users.
   - **`home.component.css`**: Styles for the home page.

4. **Login and Register Components**:
   - **`login.component.ts`** and **`register.component.ts`**: Manage user authentication and registration.
   - **`login.component.html`** and **`register.component.html`**: Templates for the login and registration forms.
   - **`login.component.css`** and **`register.component.css`**: Styles for the authentication forms.

---

### Services

1. **Auth Service**:
   - **`auth.service.ts`**: Handles user authentication, including login, logout, and registration. It interacts with the backend to validate user credentials and manage sessions.

2. **Chat Service**:
   - **`chat.service.ts`**: Manages communication between the front-end and the backend for sending and receiving chat messages. It uses HTTP requests to interact with the chatbot's API.

3. **HttpInterceptor Service**:
   - **`http-interceptor.service.ts`**: Intercepts HTTP requests and responses to handle authentication tokens, error handling, and other cross-cutting concerns.

---

### Routing
- **`app.routes.ts`**: Defines the routes for the application, enabling navigation between different components such as the home page, chat interface, and authentication forms.

---

### Assets and Styles
- **`assets`**: Contains static assets like images and icons used in the application.
- **`styles.css`**: Global styles applied across the entire application.

---

## Creating Components and Services

### Generating Components
Angular CLI is used to generate components, ensuring consistency and adherence to best practices. For example, to generate a new component:

```bash
ng generate component chat
```

This command creates the necessary files (`chat.component.ts`, `chat.component.html`, `chat.component.css`) and updates the module to include the new component.

---

### Generating Services
Services are generated similarly using Angular CLI:

```bash
ng generate service auth
```

This command creates the service file (`auth.service.ts`) and updates the module to include the service.

---
### Implementing the Chat Interface
The chat interface is implemented using Angular's data binding and event handling features.

#### `chat.component.ts`:
```typescript
import { Component, inject } from '@angular/core';
import { Router } from '@angular/router';
import { FormsModule } from '@angular/forms';
import { AuthService } from '../auth.service';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-chat',
  standalone: true,
  imports: [FormsModule, CommonModule],
  templateUrl: './chat.component.html',
  styleUrl: './chat.component.css'
})
export class ChatComponent {
  title = 'front-chatbot';
  prompt: string = '';
  loading: boolean = false;
  chatHistory: any[] = [];
  authService: AuthService = inject(AuthService);

  constructor(private router: Router) {}

  async SendData() {
    if (this.prompt) {
      this.loading = true;
      const userMessage = this.prompt;
      this.prompt = '';

      // Add user message to chat history
      this.chatHistory.push({ from: 'user', message: userMessage });

      try {
        // Call the generate_response method from AuthService
        const response = await this.authService.generate_response(userMessage).toPromise();
        
        // Add bot response to chat history
        this.chatHistory.push({ from: 'bot', message: response.response });
      } catch (error) {
        console.error('Error generating response:', error);
        this.chatHistory.push({ from: 'bot', message: 'Sorry, something went wrong. Please try again.' });
      } finally {
        this.loading = false;
      }
    }
  }

  formatText(text: string) {
    const result = text.replaceAll('*', '');
    return result;
  }

  // Navigate to home
  goToHome() {
    this.router.navigate(['/']);
  }

  // Navigate to chat (optional, since you're already on chat page)
  goToChat() {
    this.router.navigate(['/chat']);
  }

  goToChatDm() {
    this.router.navigate(['/chat-dm']);
  }

  goToChatRAG() {
    this.router.navigate(['/chat-rag']);
  }

  // Function to handle conversation model choice
  chooseModel(model: string) {
    console.log('Chosen model:', model);
    // Logic to switch chatbot model based on user's choice
  }

  logout() {
    // Clear the token from localStorage
    localStorage.removeItem('authToken');

    // Redirect to the login page
    this.router.navigate(['/home']);
  }
}
```

#### `chat.component.html`:
```html
<div class="navbar">
  <div class="logo">
    <h2>CyberSecurity Chatbot</h2>
  </div>
  <ul>
    <li><button (click)="goToHome()">Home</button></li>
    <li><button (click)="goToChat()">Chat</button></li>
    <li><button (click)="logout()">Logout</button></li>
  </ul>
</div>

<div class="chat-container">
  <header class="chat-header">
    <h4>Chatbot Fine Tuning</h4>
  </header>
  <main class="chat-history">
    <ng-container *ngFor="let item of chatHistory">
      <div class="message" [ngClass]="{'blue': item.from === 'user', 'green': item.from === 'bot'}">
        <i class="fa" [ngClass]="{'fa-user-circle': item.from === 'user', 'fa-graduation-cap': item.from === 'bot'}"></i>
        <span class="responses" [innerHTML]="formatText(item.message)"></span>
      </div>
    </ng-container>
    <ng-container *ngIf="loading">
      <div class="user-card">
        <div class="name4 skeleton"></div>
        <div class="name3 skeleton"></div>
        <div class="name2 skeleton"></div>
        <div class="name1 skeleton"></div>
      </div>
    </ng-container>
  </main>

  <footer class="chat-input">
    <input type="text" id="message-input" (keyup.enter)="SendData()" placeholder="Type your messages..." [(ngModel)]="prompt" [disabled]="loading" autocomplete="off" /> &nbsp;
    <button id="send-button" (click)="SendData()" [disabled]="loading">Send</button>
  </footer>

  <!-- Conversation choice buttons -->
  <div class="conversation-choice">
    <button (click)="goToChatRAG()">Chatbot RAG</button>
    <button (click)="goToChat()">Chatbot Fine Tuning</button>
    <button (click)="goToChatDm()">Generative AI</button>
  </div>
</div>
```

---

### Integrating with Backend Services
The front-end communicates with the backend services using Angular's `HttpClient` module. Here's an example of how the `ChatService` interacts with the backend:

#### `chat.service.ts`:
```typescript
import { EventEmitter, Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { catchError, Observable, retry, throwError, timeout } from 'rxjs';

@Injectable({
  providedIn: 'root',
})
export class AuthService {
  private apiUrl = 'http://localhost:8222/';
  authStatusChanged = new EventEmitter<void>();

  constructor(private http: HttpClient) {}

  register(userData: any): Observable<any> {
    return this.http.post(`${this.apiUrl}user/signup`, userData);
  }

  login(credentials: any): Observable<any> {
    return this.http.post(`${this.apiUrl}user/login`, credentials);
  }

  getUsers(): Observable<any> {
    return this.http.get(`${this.apiUrl}user/paginate`);
  }

  generate_response(prompt: string): Observable<any> {
    return this.http.post(`${this.apiUrl}ft/generate`, { prompt });
  }

  threat_detection(prompt: string): Observable<any> {
    const headers = new HttpHeaders({
      'Content-Type': 'application/json',
    });

    return this.http
      .post<any>(`${this.apiUrl}ai/detection`, { prompt }, { headers })
      .pipe(
        timeout(300000), // 5 minutes timeout
        retry(1),
        catchError((error) => {
          console.error('API Error:', error);
          throw error;
        })
      );
  }

  query_rag(prompt: string): Observable<any> {
    return this.http.post(`${this.apiUrl}rag/generate`, { query_text: prompt });
  }

  saveToken(token: string): void {
    localStorage.setItem('authToken', token);
  }

  getToken(): string | null {
    return localStorage.getItem('authToken');
  }

  removeToken() {
    localStorage.removeItem('token');
    this.authStatusChanged.emit();
  }
}
```

## User Interface Screenshots
After integrating the front-end with the backend, the user interface provides a seamless experience for interacting with the chatbot. Below are some screenshots of the application in action:

### 1.Home PageThe 
home page welcomes users and provides an option to start a secure conversation with the chatbot. It features a clean and intuitive design with a navigation bar at the top, including options for Home, Chat, Register, and Login. The center of the page displays a welcoming message, "Welcome to CyberSecurity Awareness", along with a call-to-action button labeled "Start Secure Conversation". This button invites users to begin interacting with the chatbot, emphasizing the application's goal of enhancing cybersecurity knowledge through AI-driven conversations. The design is user-friendly, encouraging users to either register, log in, or directly engage with the chatbot to learn more about cybersecurity in an interactive way.
![ih2](https://github.com/user-attachments/assets/a154ce8d-7143-425e-879e-5ca444b2222b)
### 1.Chat Interface
L'interface de chat permet aux utilisateurs d'interagir avec le chatbot. Les utilisateurs peuvent saisir des messages, et le chatbot répond en temps réel. L'interface comprend une barre de navigation similaire à celle de la page d'accueil, un en-tête intitulé "Chatbot Fine Tuning", et une section d'historique de chat affichant les messages des utilisateurs et les réponses du chatbot. Les utilisateurs peuvent saisir leurs requêtes dans le champ de saisie intitulé "type your messages..." et les envoyer en utilisant le bouton Send. De plus, l'interface propose des options pour basculer entre différents modes de chatbot : Chatbot RAG, Chatbot Fine Tuning, et Generative AI, permettant aux utilisateurs d'adapter leur expérience en fonction de leurs besoins. Le design est convivial et axé sur une interaction fluide avec le chatbot.
![ih1](https://github.com/user-attachments/assets/d7bf6c91-2b3c-43c6-aa4a-6518932970bb)

---

## Conclusion
This project successfully combines advanced machine learning technologies and microservices architecture to develop a sophisticated cybersecurity-focused chatbot. The chatbot leverages powerful Language Understanding Models (LUMs), Diffusion Models for threat forecasting, and Retrieval-Augmented Generation (RAG) to deliver intelligent and relevant responses tailored to cybersecurity needs.

By adopting a microservices approach, the system ensures scalability, flexibility, and maintainability. Integrating best practices in DevOps and MLOps further supports continuous development, deployment, and monitoring.

The project demonstrated effective collaboration among various roles, from front-end development with Angular to backend implementation, CI/CD pipelines, security enhancements, and specialized responsibilities in MLOps and model training. Technologies like Spring Boot, Flask, Angular, and Nginx played pivotal roles in building, hosting, and deploying the services.
