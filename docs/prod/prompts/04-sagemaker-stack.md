# Prompt 04 — SagemakerStack

> Prerequisite: prompt 03 done (glmocr-sglang image in ECR, weights in S3, SSM model_data_key set). Expect ~30 min + 10-min provisioning wait.

---

```
Implement SagemakerStack in internal/stacks/sagemaker.go.

Read first:
  - docs/prod/03-sagemaker-sglang-byoc.md (full)
  - docs/prod/07-secrets-and-config.md (for SSM knob wiring)

Tasks:

1. IAM: execution role glmocr-sglang-sagemaker-role
   - Trust: sagemaker.amazonaws.com
   - Policies:
     * Grants from the weights S3 bucket (Read on the whole prefix)
     * AWSSageMakerExecutionRolePolicy (managed) — for CW log/metric writes
   Export as SagemakerExecutionRole.

2. CfnModel (awssagemaker):
   - ModelName: glmocr-sglang
   - ExecutionRoleArn: the role above
   - PrimaryContainer:
     * Image: from EcrStack.SglangRepository + cfg.SglangImageTag
     * ModelDataUrl: built from the S3 bucket + SSM param (read at synth via
       awsssm.StringParameter_ValueFromLookup — cache via cdk.context.json)
     * Environment: ALL the SGL_* knobs. Read each from SSM via
       ValueFromLookup on "/glmocr/prod/sgl/<KEY>". Create these SSM params
       now if they don't exist (inline CfnParameter or a separate SSMSeeder
       construct). Seed with the dev-tuned defaults from reference/env-tuned.md.

     Do NOT set SGLANG_USE_CUDA_IPC_TRANSPORT in the first deploy.

3. CfnEndpointConfig:
   - EndpointConfigName: glmocr-sglang-cfg-<sha>  (include cfg.SglangImageTag to
     force a new config on each image change — blue-green)
   - ProductionVariants: one, with:
     * InitialInstanceCount: 1 (from cfg.sagemakerMinInstances)
     * InstanceType: ml.g4dn.2xlarge (from cfg.sagemakerInstance)
     * InitialVariantWeight: 1
     * ModelDataDownloadTimeoutInSeconds: 600
     * ContainerStartupHealthCheckTimeoutInSeconds: 1800
   - DeploymentConfig:
     * BlueGreenUpdatePolicy:
         TrafficRoutingConfiguration: ALL_AT_ONCE with a 5-minute bake
         MaximumExecutionTimeoutInSeconds: 1800
     * AutoRollbackConfiguration:
         Alarms: reference a CloudWatch alarm on `Invocation5XXErrors` > 5 in 1 min

4. CfnEndpoint:
   - EndpointName: glmocr-sglang
   - EndpointConfigName: (above)

5. Export:
   - EndpointName  (string)
   - EndpointArn   (for IAM Grant)
   - GrantInvoke(role iam.IRole)  — attaches sagemaker:InvokeEndpoint on this arn

6. Tests:
   - Model + EndpointConfig + Endpoint all present
   - The Environment block contains at least SGL_SCHEDULE_POLICY=lpm and
     SGL_SPECULATIVE=true (catch accidental removal of tuned knobs)

7. Deploy:
     cdk deploy --context stage=prod glmocr-sagemaker-prod
   Endpoint creation takes 8-15 min. Tail logs:
     aws logs tail /aws/sagemaker/Endpoints/glmocr-sglang/AllTraffic --follow
   Expect to see SGLang's own startup output (CUDA init, weight load, server on :30000)
   followed by uvicorn coming up on :8080.

8. Verify:
     aws sagemaker describe-endpoint --endpoint-name glmocr-sglang --query EndpointStatus
   should print "InService".

     aws sagemaker-runtime invoke-endpoint \
       --endpoint-name glmocr-sglang \
       --content-type application/json \
       --body '{"model":"glm-ocr","messages":[{"role":"user","content":"hello"}],"max_tokens":16}' \
       /tmp/resp.json
     cat /tmp/resp.json   # should be an OpenAI chat-completions shape

Commit as "feat(sagemaker): BYOC sglang endpoint + blue-green deploy config".

Report back:
  - EndpointName
  - Time taken to become InService
  - Any OOM or other concerning log lines (share first few)
```
