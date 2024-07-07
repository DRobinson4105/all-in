#!/bin/bash

download_dir="mistral"

mkdir -p "$download_dir"

urls=(
    "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/resolve/main/config.json"
    "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/resolve/main/consolidated.safetensors"
    "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/resolve/main/generation_config.json"
    "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/resolve/main/model-00001-of-00003.safetensors"
    "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/resolve/main/model-00002-of-00003.safetensors"
    "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/resolve/main/model-00003-of-00003.safetensors"
    "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/resolve/main/model.safetensors.index.json"
    "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/resolve/main/params.json"
    "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/resolve/main/special_tokens_map.json"
    "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/resolve/main/tokenizer.json"
    "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/resolve/main/tokenizer.model"
    "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/resolve/main/tokenizer.model.v3"
    "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/resolve/main/tokenizer_config.json"
)

login_url="https://huggingface.co/login"

prompt_credentials() {
    read -p "Enter your Hugging Face username or email: " username
    read -sp "Enter your Hugging Face password: " password
    echo
}

perform_login() {
    login_response=$(curl -s -o /dev/null -w "%{http_code}" -c "$cookie_file" -d "username=$username&password=$password" "$login_url")
}

cookie_file=$(mktemp)

prompt_credentials

perform_login

while [ "$login_response" -ne 200 ]; do
    echo "Invalid username or password. Please try again."
    prompt_credentials
    perform_login
done

echo "Login successful."

for url in "${urls[@]}"; do
    filename=$(basename "$url")
    
    curl -L -b "$cookie_file" -o "$download_dir/$filename" "$url"
    
    echo "Downloaded $url to $download_dir/$filename"
done

rm "$cookie_file"

echo "All files have been downloaded."