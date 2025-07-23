FROM prom/prometheus:v2.37.0

COPY prometheus.yml /etc/prometheus/
COPY prometheus.config /etc/prometheus/

EXPOSE 9090

ENTRYPOINT [ "/bin/prometheus" ]
CMD [ "--config.file=/etc/prometheus/prometheus.yml", \
    "--storage.tsdb.path=/prometheus", \
    "--web.console.libraries=/usr/share/prometheus/console_libraries", \
    "--web.console.templates=/usr/share/prometheus/consoles" ] 