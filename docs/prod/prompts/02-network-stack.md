# Prompt 02 — NetworkStack (VPC + endpoints)

> Prerequisite: prompt 01 complete (`cdk synth` passes). Expect ~30 min.

---

```
Implement NetworkStack in internal/stacks/network.go.

Read first:
  - docs/prod/01-prod-architecture.md (VPC section)
  - docs/prod/02-cdk-go-structure.md

Requirements:

1. VPC:
   - Name: glmocr-prod
   - CIDR: 10.42.0.0/16 (ask if you want different)
   - 2 AZs (cost-conscious; scale to 3 later)
   - Subnets: 2 public + 2 private (Isolated is fine — we use VPC endpoints)
   - NAT gateway: 1 (single-AZ, shared). In prod HA we'd want 2, but the cost
     doubles; flag this as a future upgrade in a code comment.

2. VPC endpoints (all Gateway or Interface, in private subnets):
   - sagemaker-runtime (Interface)  — CPU task → SageMaker InvokeEndpoint
   - s3 (Gateway)                   — weights pull, reports upload
   - ecr.api (Interface)            — image pull
   - ecr.dkr (Interface)            — image pull (layers)
   - secretsmanager (Interface)     — Secrets Manager reads
   - ssm (Interface)                — Parameter Store reads
   - logs (Interface)               — CloudWatch log writes
   - monitoring (Interface)         — CloudWatch metrics (alarm evaluation)

   For each: enable privateDnsEnabled, open security groups to VPC CIDR on 443.

3. Export (construct field accessors — no raw strings across stacks):
   - Vpc
   - PublicSubnets, PrivateSubnets
   - SecurityGroupForFargate (allow inbound 5002 from ALB SG only)
   - SecurityGroupForAlb (allow inbound 5002 from VPC CIDR — it's an internal ALB)

4. Tag every resource with:
   - Project: glmocr
   - Stage:   prod
   - ManagedBy: cdk

5. Tests: internal/stacks/network_test.go
   - VPC has 2 AZs × 2 subnets each
   - All 8 endpoints present
   - NAT gateway exists
   Use cdk assertions: Template_FromStack + HasResource.

Run `cdk synth --context stage=prod glmocr-network-prod` to verify before
deploy.

Deploy:
   cdk deploy --context stage=prod glmocr-network-prod
Expect ~5 minutes.

After success:
  - aws ec2 describe-vpcs --filters Name=tag:Project,Values=glmocr
  - Confirm VPC CIDR matches.
  - Screenshot or note the VPC ID.

Commit as "feat(network): VPC + 8 VPC endpoints".

Do NOT proceed to the next stack until deploy is green.
```
