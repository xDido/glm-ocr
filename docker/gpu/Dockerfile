FROM lmsysorg/sglang:latest

WORKDIR /app

COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

EXPOSE 30000

# Invoke via `bash` explicitly (PATH lookup) so we don't depend on the
# script's exec bit surviving a Windows checkout, or on the shebang being
# parseable when the file picks up CRLF line endings. Same rationale as the
# sglang service in docker-compose.yml.
ENTRYPOINT ["bash", "/app/entrypoint.sh"]
