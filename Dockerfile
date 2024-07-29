FROM python:3.11.5

# We create an /opt directory
RUN set -x \
    && mkdir /opt/voodoo_homework

# Now that we've created our virtual environment, we'll go ahead and update
# our $PATH to refer to it first
ENV PATH="/opt/voodoo_homework/bin:${PATH}"
ENV PYTHONUNBUFFERED 1

# Next, we want to update pip, setuptools, and wheel inside of this virtual
# environment to ensure that we have the latest versions of them.
RUN pip --no-cache-dir --disable-pip-version-check install --upgrade -q pip setuptools wheel

COPY . /opt/voodoo_homework

WORKDIR /opt/voodoo_homework

# Install the Python requirements, this is done after copying the requirements
# but prior to copying our code into the container so that changes don't require
# triggering an entire install of all of our dependencies.
RUN pip install -e .

ENTRYPOINT ["python", "-m", "voodoo_homework"]
