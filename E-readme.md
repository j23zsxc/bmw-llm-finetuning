# huggingface dataset link
https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k/tree/main



# 假设你的 Docker Hub 用户名是 'your-dockerhub-username'
# 并且你想标记镜像为 'qwen-finetune' 版本 'v1.0'
docker build --platform linux/amd64 -t qwen-finetune:v1.0 .


## 推送到ECR
aws ecr get-login-password --region eu-central-1 | docker login --username AWS --password-stdin 974677654443.dkr.ecr.eu-central-1.amazonaws.com

docker tag qwen-finetune:v1.0 974677654443.dkr.ecr.eu-central-1.amazonaws.com/ai-lab/r2d2/finetune:v1.0

docker push 974677654443.dkr.ecr.eu-central-1.amazonaws.com/ai-lab/r2d2/finetune:v1.0
