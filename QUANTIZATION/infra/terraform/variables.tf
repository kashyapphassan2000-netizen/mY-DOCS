variable "region" {
  description = "AWS region with GPU capacity (us-east-1 has the most)"
  type        = string
  default     = "us-east-1"
}

variable "cluster_name" {
  type    = string
  default = "llm-serving"
}

variable "kubernetes_version" {
  description = "Pick the newest version shown in the EKS console (older versions cost extra support fees)"
  type        = string
  default     = "1.34"
}

variable "gpu_instance_type" {
  description = "g6e.xlarge = 1x L40S 48 GB; g6.xlarge = 1x L4 24 GB; g6e.12xlarge = 4x L40S"
  type        = string
  default     = "g6e.xlarge"
}

variable "gpu_min" {
  description = "GPU machines that are always on"
  type        = number
  default     = 1
}

variable "gpu_max" {
  description = "Hard ceiling on GPU machines (your 50-machine limit). Needs an AWS quota increase."
  type        = number
  default     = 50
}
