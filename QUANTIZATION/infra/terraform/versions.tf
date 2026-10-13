terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.95, < 6.0"
    }
  }
}

provider "aws" {
  region = var.region
  default_tags {
    tags = { project = var.cluster_name, managed_by = "terraform" }
  }
}
