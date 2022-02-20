FROM manjarolinux/base

RUN pacman-mirrors --fasttrack 
RUN yes | pacman --noconfirm -S python-pip 

RUN pip install --upgrade pip
RUN pip3 install gunicorn django sklearn pandas tensorflow matplotlib

COPY . .

EXPOSE 8080

CMD ["gunicorn", "api_modelos.wsgi:application", "--bind", "0.0.0.0:8080"]