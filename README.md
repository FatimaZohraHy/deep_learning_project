# Cybersecurity-Focused Chatbot Project

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

### Front-end Developer (Angular / React JS)
- Develop the chatbot's user interface using **Angular** or **React JS**.
- Ensure that the frontend is intuitive and responsive for security analysts to interact with the chatbot seamlessly.
- Integrate the frontend with the backend APIs for real-time communication.

### Backend / CI/CD / Security
- Implement backend services to handle business logic and data processing.
- Design and manage the **CI/CD pipelines** to automate the testing, building, and deployment of the chatbot's microservices.
- Ensure the security of the backend systems and integrate security measures, including encryption and secure API handling.

### LLMs Responsible / MLOps
As the LLMs responsible, my role involves optimizing and fine-tuning pre-trained Language Understanding Models in our case we used the **llama3-8b** to ensure they accurately handle cybersecurity-related queries. This process is essential for adapting the models to our specific domain and use case. The fine-tuning steps include the following:
## 1. Fine-Tuning the Model
![image](https://github.com/user-attachments/assets/d5a60941-7f2c-4932-bb95-b7d0994efdf5)

   Fine-tuning is a critical step to adapt the pre-trained models to the specific needs of cybersecurity-related question answering. The process involves the following steps:
   - **Step 1: Preparing the Dataset**
   To fine-tune the models, we start by collecting and structuring a dataset tailored to cybersecurity. The dataset contains cybersecurity-related question-answer pairs, formatted in a way that allows the model to learn from previous interactions and respond with relevant, domain-specific knowledge.
we also used **generated synthetic data** using LLMs you can find it here: **_https://huggingface.co/datasets/Joussef/my_sec_data_**
- **Step 2: Loading Pre-Trained Models**
   We begin by loading the pre-trained model, using a specialized framework like **_unsloth_** or **_transformers_**. This provides a strong foundation for the fine-tuning process since these models have already been trained on a large corpus of text data. By using these models, we can leverage their existing language capabilities while specializing them for cybersecurity tasks.
- **Step 3: Fine-Tuning the Model**
   The fine-tuning process involves training the model on our custom dataset. We adjust hyperparameters like batch size, learning rate, and number of epochs to optimize the model's performance. During this step, the model learns how to respond accurately to cybersecurity-related queries, improving its ability to understand and generate relevant answers.
- **Step 4: Evaluation and Testing**
   After fine-tuning, the model is tested using a validation dataset to evaluate its performance. We look at various metrics such as accuracy, relevance, and response time to ensure that the model is responding correctly to cybersecurity questions.
- **Step 5: Saving and Deploying the Fine-Tuned Model**
   Once the model is fine-tuned and evaluated, we save it to a storage location. The fine-tuned model is then ready for deployment, where it will be integrated into the cybersecurity chatbot or other applications that require cybersecurity expertise.

## 2. Retrieval-Augmented Generation (RAG)
Retrieval-Augmented Generation (RAG) is a powerful architecture that enhances the ability of language models to generate accurate responses by combining a retrieval mechanism with generative capabilities. In this project, RAG is used to build a cybersecurity-focused chatbot capable of answering domain-specific queries by retrieving relevant context from a document database.
- **Step 1: Embedding Function**
- The embedding function is a critical component of RAG. It maps both documents and user queries into a shared vector space, enabling efficient similarity-based search. In this project, a pre-trained embedding model optimized for semantic understanding is used. This ensures that document embeddings capture the nuanced meanings necessary for accurate retrieval.
  - **Purpose**: Convert text data into numerical representations.
  - **Key Advantage**: Enables fast and meaningful comparisons between query and document content.
- **Step 2: Populating the Database**
- This step involves preparing the document database for retrieval. It consists of several sub-steps:
  1. Loading Documents:
     - Documents are imported from a directory containing PDF files.
     - This allows seamless integration of real-world documents into the system.
  2. Splitting Text into Chunks:
     - To optimize retrieval, documents are divided into smaller chunks of text
     - Overlapping chunks ensure no loss of context while maintaining manageable sizes for processing.
  3. Storing Embeddings in a Vector Database:
     - The text chunks are embedded into a vector database, ensuring that they can be retrieved based on their semantic similarity to user queries.
     - A persistent database is used to retain embeddings between sessions.
  4. Metadata Assignment:
     - Unique identifiers are created for each text chunk based on its source document and position.
     - This metadata ensures traceability and efficient retrieval of information.
- **Step 3: Querying the Database**
- Once the database is populated, it can be queried to retrieve relevant context for answering user questions. The workflow for querying involves:
  1. Retrieving Context:
     - The system uses a similarity-based search to find the most relevant text chunks from the database.
     - The top results are combined to form a comprehensive context.
  2. Generating a Response:
     - The retrieved context is used to generate a tailored response to the user query.
     - A fine-tuned language model is employed to ensure that the generated answers are accurate and domain-specific.
  3. Providing Sources:
     - Alongside the generated response, the system also outputs the source documents used in the answer.
     - This enhances transparency and trustworthiness.

### Diffusion Model Responsible / MLOps
- Develop and deploy **Diffusion Models** that simulate the spread of threats and vulnerabilities across networks.
- Integrate diffusion models with the chatbot's backend to predict future security risks based on historical data.
- Apply MLOps best practices for model management, including continuous training, monitoring, and versioning.

### DevOps Responsible / Monitoring
- Implement **DevOps** practices for infrastructure automation, continuous integration, and continuous deployment.
- Use tools like **Docker**, **Kubernetes**, and **Terraform** to containerize and deploy microservices.
- Set up monitoring using **Prometheus**, **Grafana**, and **ELK Stack** to track system performance and ensure uptime.

## Technologies Used

- **Frontend**: Angular, React JS
- **Backend**: Node.js, Python (Flask/Django for APIs)
- **Language Models**: GPT-4, BERT
- **Diffusion Models**: Custom or pre-built models for threat diffusion forecasting
- **CI/CD**: Jenkins, GitLab CI, or GitHub Actions
- **Containerization**: Docker, Kubernetes
- **Infrastructure as Code**: Terraform, Ansible
- **Monitoring and Logging**: Prometheus, Grafana, ELK Stack
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
## 3. Authentication Service using JWT

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


# Generative AI Service

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





# RAG Flask Service

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

---

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
