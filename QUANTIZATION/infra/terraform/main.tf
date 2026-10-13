data "aws_availability_zones" "available" {
  state = "available"
}

locals {
  azs = slice(data.aws_availability_zones.available.names, 0, 2)
}

# 1. Network: a private network (VPC) with public subnets (load balancer) and private subnets (GPU machines)
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name            = "${var.cluster_name}-vpc"
  cidr            = "10.0.0.0/16"
  azs             = local.azs
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24"]

  enable_nat_gateway = true # private machines can download models from Hugging Face
  single_nat_gateway = true # one NAT = cheaper; use one per AZ for production HA

  public_subnet_tags  = { "kubernetes.io/role/elb" = 1 }
  private_subnet_tags = { "kubernetes.io/role/internal-elb" = 1 }
}

# 2. Permission for the cluster-autoscaler to add/remove GPU machines
resource "aws_iam_policy" "cluster_autoscaler" {
  name = "${var.cluster_name}-cluster-autoscaler"
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "autoscaling:DescribeAutoScalingGroups",
        "autoscaling:DescribeAutoScalingInstances",
        "autoscaling:DescribeLaunchConfigurations",
        "autoscaling:DescribeScalingActivities",
        "autoscaling:DescribeTags",
        "autoscaling:SetDesiredCapacity",
        "autoscaling:TerminateInstanceInAutoScalingGroup",
        "ec2:DescribeImages",
        "ec2:DescribeInstanceTypes",
        "ec2:DescribeLaunchTemplateVersions",
        "ec2:GetInstanceTypesFromInstanceRequirements",
        "eks:DescribeNodegroup"
      ]
      Resource = "*"
    }]
  })
}

# 3. Kubernetes cluster (EKS) with two machine pools
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.0"

  cluster_name                             = var.cluster_name
  cluster_version                          = var.kubernetes_version
  cluster_endpoint_public_access           = true # kubectl from your laptop
  enable_cluster_creator_admin_permissions = true # you are admin of the cluster you create

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  cluster_addons = {
    coredns                = {}
    kube-proxy             = {}
    vpc-cni                = {}
    eks-pod-identity-agent = {}
  }

  eks_managed_node_groups = {
    # small CPU machines for Prometheus, KEDA, cluster-autoscaler
    system = {
      ami_type       = "AL2023_x86_64_STANDARD"
      instance_types = ["m6i.xlarge"] # 4 vCPU / 16 GB: Prometheus needs room
      min_size       = 2
      max_size       = 3
      desired_size   = 2
      iam_role_additional_policies = {
        autoscaler = aws_iam_policy.cluster_autoscaler.arn
      }
    }

    # GPU machines for vLLM. NVIDIA AMI = drivers pre-installed.
    gpu = {
      ami_type       = "AL2023_x86_64_NVIDIA"
      instance_types = [var.gpu_instance_type]
      min_size       = var.gpu_min
      max_size       = var.gpu_max
      desired_size   = var.gpu_min

      block_device_mappings = {
        xvda = {
          device_name = "/dev/xvda"
          ebs = {
            volume_size           = 200 # model download + container image
            volume_type           = "gp3"
            delete_on_termination = true
          }
        }
      }

      labels = {
        workload                 = "llm-gpu"
        "nvidia.com/gpu.present" = "true" # the NVIDIA device-plugin Helm chart only runs on nodes with this label
      }
      taints = {
        gpu = { key = "nvidia.com/gpu", value = "true", effect = "NO_SCHEDULE" } # only GPU pods land here
      }
    }
  }
}
