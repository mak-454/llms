cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: streamlit-webapp-llms
  namespace: default
spec:
  replicas: 1
  selector:
    matchLabels:
      app: streamlit-webapp-llms
  template:
    metadata:
       labels:
         app: streamlit-webapp-llms
    spec:
      containers:
      - name: app
        image: ocdr/llms:webapp
        imagePullPolicy: Always

EOF

cat <<EOF | kubectl apply -f -
kind: Service
apiVersion: v1
metadata:
  name: streamlit-webapp-llms
  namespace: default
spec:
  selector:
    app: streamlit-webapp-llms
  ports:
  - protocol: TCP
    port: 8501
    nodePort: 31555
  type: NodePort

EOF
sleep 30s
echo "streamlit webapp is available @ <dkube-ip>:31444"

