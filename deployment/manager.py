from typing import Dict, List, Optional, Any
from pathlib import Path
import logging
import yaml
import docker
import kubernetes
import requests




class DeploymentManager:
    """Manage model deployment and environment"""
    def __init__(self,
                 config_path: str,
                 environment: str = "production"):
        self.config = self._load_config(config_path)
        self.environment = environment
        
        # Setup logging
        self.logger = logging.getLogger("deployment")
        self._setup_logging()
        
        # Initialize components
        self.docker_client = docker.from_env()
        self.k8s_config = kubernetes.config.load_kube_config()
        self.k8s_client = kubernetes.client.CoreV1Api()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load deployment configuration"""
        with open(config_path) as f:
            config = yaml.safe_load(f)
            
        # Validate configuration
        required_keys = ['model_version', 'docker_image', 'resource_requirements']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")
                
        return config
        
    def _setup_logging(self):
        """Setup deployment logging"""
        log_dir = Path("logs/deployment")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        handler = logging.FileHandler(log_dir / f"{self.environment}.log")
        handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        self.logger.addHandler(handler)
        
    def deploy(self) -> bool:
        """Run deployment pipeline"""
        try:
            # Check environment
            if not self._check_environment():
                return False
                
            # Build and push Docker image
            self._build_docker_image()
            
            # Deploy to Kubernetes
            self._deploy_to_kubernetes()
            
            # Verify deployment
            if not self._verify_deployment():
                self._rollback_deployment()
                return False
                
            self.logger.info("Deployment completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Deployment failed: {str(e)}")
            self._rollback_deployment()
            return False
            
    def _check_environment(self) -> bool:
        """Check deployment environment"""
        # Check resource availability
        available_memory = self._get_available_memory()
        required_memory = self.config['resource_requirements']['memory']
        
        if available_memory < required_memory:
            self.logger.error(f"Insufficient memory: {available_memory}GB < {required_memory}GB")
            return False
            
        # Check GPU availability if required
        if self.config['resource_requirements'].get('gpu', False):
            if not self._check_gpu_availability():
                return False
                
        # Check disk space
        available_disk = self._get_available_disk_space()
        required_disk = self.config['resource_requirements']['disk']
        
        if available_disk < required_disk:
            self.logger.error(f"Insufficient disk space: {available_disk}GB < {required_disk}GB")
            return False
            
        return True
        
    def _build_docker_image(self):
        """Build and push Docker image"""
        self.logger.info("Building Docker image...")
        
        # Build image
        image, build_logs = self.docker_client.images.build(
            path=".",
            tag=self.config['docker_image'],
            rm=True,
            forcerm=True
        )
        
        # Log build output
        for chunk in build_logs:
            if 'stream' in chunk:
                self.logger.info(chunk['stream'].strip())
                
        # Push image
        self.logger.info("Pushing Docker image...")
        for line in self.docker_client.images.push(
            self.config['docker_image'],
            stream=True,
            decode=True
        ):
            if 'status' in line:
                self.logger.info(line['status'])
                
    def _deploy_to_kubernetes(self):
        """Deploy to Kubernetes cluster"""
        self.logger.info("Deploying to Kubernetes...")
        
        # Create deployment
        deployment = self._create_deployment_manifest()
        
        try:
            # Apply deployment
            api = kubernetes.client.AppsV1Api()
            api.create_namespaced_deployment(
                body=deployment,
                namespace=self.environment
            )
            
            # Create service
            service = self._create_service_manifest()
            self.k8s_client.create_namespaced_service(
                body=service,
                namespace=self.environment
            )
            
        except kubernetes.client.rest.ApiException as e:
            self.logger.error(f"Kubernetes API error: {str(e)}")
            raise
            
    def _verify_deployment(self) -> bool:
        """Verify deployment status"""
        self.logger.info("Verifying deployment...")
        
        try:
            # Check pod status
            pods = self.k8s_client.list_namespaced_pod(
                namespace=self.environment,
                label_selector=f"app={self.config['app_name']}"
            )
            
            for pod in pods.items:
                if pod.status.phase != 'Running':
                    self.logger.error(f"Pod {pod.metadata.name} not running: {pod.status.phase}")
                    return False
                    
            # Check service availability
            service = self.k8s_client.read_namespaced_service(
                name=self.config['service_name'],
                namespace=self.environment
            )
            
            if not service.status.load_balancer.ingress:
                self.logger.error("Service not available")
                return False
                
            # Run health checks
            if not self._run_health_checks():
                return False
                
            return True
            
        except kubernetes.client.rest.ApiException as e:
            self.logger.error(f"Verification error: {str(e)}")
            return False
            
    def _rollback_deployment(self):
        """Rollback deployment on failure"""
        self.logger.info("Rolling back deployment...")
        
        try:
            # Delete new deployment
            api = kubernetes.client.AppsV1Api()
            api.delete_namespaced_deployment(
                name=self.config['app_name'],
                namespace=self.environment
            )
            
            # Delete service
            self.k8s_client.delete_namespaced_service(
                name=self.config['service_name'],
                namespace=self.environment
            )
            
            # Restore previous version if available
            if self.config.get('previous_version'):
                self._deploy_version(self.config['previous_version'])
                
        except kubernetes.client.rest.ApiException as e:
            self.logger.error(f"Rollback error: {str(e)}")
            
    def _deploy_version(self, version: str):
        """Deploy specific version"""
        self.config['model_version'] = version
        self._deploy_to_kubernetes()
        
    def _create_deployment_manifest(self) -> dict:
        """Create Kubernetes deployment manifest"""
        return {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': self.config['app_name'],
                'namespace': self.environment
            },
            'spec': {
                'replicas': self.config.get('replicas', 1),
                'selector': {
                    'matchLabels': {
                        'app': self.config['app_name']
                    }
                },
                'template': {
                    'metadata': {
                        'labels': {
                            'app': self.config['app_name']
                        }
                    },
                    'spec': {
                        'containers': [{
                            'name': self.config['app_name'],
                            'image': self.config['docker_image'],
                            'resources': {
                                'requests': self.config['resource_requirements'],
                                'limits': self.config['resource_limits']
                            },
                            'ports': [{
                                'containerPort': 8000
                            }],
                            'env': self.config.get('environment_variables', []),
                            'readinessProbe': {
                                'httpGet': {
                                    'path': '/health',
                                    'port': 8000
                                },
                                'initialDelaySeconds': 5,
                                'periodSeconds': 10
                            }
                        }]
                    }
                }
            }
        }
        
    def _create_service_manifest(self) -> dict:
        """Create Kubernetes service manifest"""
        return {
            'apiVersion': 'v1',
            'kind': 'Service',
            'metadata': {
                'name': self.config['service_name'],
                'namespace': self.environment
            },
            'spec': {
                'selector': {
                    'app': self.config['app_name']
                },
                'ports': [{
                    'port': 80,
                    'targetPort': 8000
                }],
                'type': 'LoadBalancer'
            }
        }
        
    def _run_health_checks(self) -> bool:
        """Run deployment health checks"""
        self.logger.info("Running health checks...")
        
        checks = [
            self._check_pod_health,
            self._check_service_health,
            self._check_model_health
        ]
        
        for check in checks:
            if not check():
                return False
                
        return True
        
    def _check_pod_health(self) -> bool:
        """Check pod health metrics"""
        try:
            metrics_api = kubernetes.client.CustomObjectsApi()
            pod_metrics = metrics_api.list_namespaced_custom_object(
                group="metrics.k8s.io",
                version="v1beta1",
                namespace=self.environment,
                plural="pods",
                label_selector=f"app={self.config['app_name']}"
            )
            
            for pod in pod_metrics['items']:
                # Check CPU usage
                cpu_usage = pod['containers'][0]['usage']['cpu']
                if self._parse_cpu(cpu_usage) > self.config['resource_limits']['cpu']:
                    self.logger.error(f"High CPU usage: {cpu_usage}")
                    return False
                    
                # Check memory usage
                memory_usage = pod['containers'][0]['usage']['memory']
                if self._parse_memory(memory_usage) > self.config['resource_limits']['memory']:
                    self.logger.error(f"High memory usage: {memory_usage}")
                    return False
                    
            return True
            
        except Exception as e:
            self.logger.error(f"Pod health check error: {str(e)}")
            return False
            
    def _check_service_health(self) -> bool:
        """Check service health"""
        try:
            # Get service endpoint
            service = self.k8s_client.read_namespaced_service(
                name=self.config['service_name'],
                namespace=self.environment
            )
            
            endpoint = service.status.load_balancer.ingress[0].ip
            
            # Check endpoint health
            response = requests.get(f"http://{endpoint}/health")
            return response.status_code == 200
            
        except Exception as e:
            self.logger.error(f"Service health check error: {str(e)}")
            return False
            
    def _check_model_health(self) -> bool:
        """Check model inference health"""
        try:
            # Get service endpoint
            service = self.k8s_client.read_namespaced_service(
                name=self.config['service_name'],
                namespace=self.environment
            )
            
            endpoint = service.status.load_balancer.ingress[0].ip
            
            # Run test inference
            test_input = "This is a test input."
            response = requests.post(
                f"http://{endpoint}/predict",
                json={'text': test_input}
            )
            
            if response.status_code != 200:
                self.logger.error(f"Model inference failed: {response.status_code}")
                return False
                
            # Validate response format
            try:
                result = response.json()
                if not all(k in result for k in ['prediction', 'confidence']):
                    self.logger.error("Invalid response format")
                    return False
            except ValueError:
                self.logger.error("Invalid JSON response")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Model health check error: {str(e)}")
            return False
            
    def _parse_cpu(self, cpu_str: str) -> float:
        """Parse CPU string to number"""
        if cpu_str.endswith('m'):
            return float(cpu_str[:-1]) / 1000
        return float(cpu_str)
        
    def _parse_memory(self, memory_str: str) -> int:
        """Parse memory string to bytes"""
        units = {'Ki': 1024, 'Mi': 1024**2, 'Gi': 1024**3}
        for unit, multiplier in units.items():
            if memory_str.endswith(unit):
                return int(memory_str[:-2]) * multiplier
        return int(memory_str)