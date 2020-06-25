FROM mrartiemea-nlap-docker-local.jfrog.io/analytics_scheduler_base:20200313.1

COPY config ./config
COPY data ./data
COPY models ./models
COPY scoring ./scoring
COPY standard_metric ./standard_metric

COPY health_expand_case.py ./health_expand_case.py
COPY health_expand_scoring_wrapper.py ./health_expand_scoring_wrapper.py
COPY health_expand_std_metrics_wrapper.py ./health_expand_std_metrics_wrapper.py
COPY health_expand_train_wrapper.py ./health_expand_train_wrapper.py

COPY startup.sh /opt/startup/bin/startup.sh

RUN chmod 770 /opt/startup/bin/startup.sh
#CMD ["/bin/bash"]
CMD ["bash", "-c", "/opt/startup/bin/startup.sh"]