# Docker Example Project

This project demonstrates how to set up a simple application using Docker. Below are the instructions for building and running the application.

## Project Structure

```
docker-example
├── src
│   ├── app.py
│   └── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .env.example
├── Makefile
└── README.md
```

## Prerequisites

- Docker installed on your machine
- Docker Compose installed

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd docker-example
   ```

2. **Build the Docker image:**
   ```
   docker build -t docker-example .
   ```

3. **Run the application using Docker Compose:**
   ```
   docker-compose up
   ```

4. **Access the application:**
   Open your web browser and go to `http://localhost:5000` (or the port specified in your `docker-compose.yml`).

## Environment Variables

You can customize the application settings by creating a `.env` file based on the `.env.example` file provided.

## Makefile Commands

You can use the Makefile to simplify common tasks. Here are some commands you can use:

- `make build` - Build the Docker image
- `make up` - Start the application
- `make down` - Stop the application

## License

This project is licensed under the MIT License. See the LICENSE file for more details.