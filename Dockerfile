FROM python:3.6-slim



RUN groupadd -r evaluator && useradd -m --no-log-init -r -g evaluator evaluator

RUN mkdir -p /opt/evaluation /input /output \
    && chown evaluator:evaluator /opt/evaluation /input /output

USER evaluator
WORKDIR /opt/evaluation

ENV PATH="/home/evaluator/.local/bin:${PATH}"

RUN python -m pip install --user -U pip

COPY --chown=evaluator:evaluator ground-truth /opt/evaluation/ground-truth

COPY --chown=evaluator:evaluator requirements.txt /opt/evaluation/
RUN python -m pip install --user -r requirements.txt

COPY --chown=evaluator:evaluator evaluation.py /opt/evaluation/

COPY --chown=evaluator:evaluator metrics_utils.py /opt/evaluation/

ENTRYPOINT "python" "-m" "evaluation"
