FROM lmsysorg/sglang:latest-cu130-runtime

WORKDIR /app

COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

EXPOSE 30000

ENTRYPOINT ["/app/entrypoint.sh"]
