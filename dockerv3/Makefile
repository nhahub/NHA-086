# Makefile for Docker Example Project

.PHONY: build run stop clean

build:
	docker-compose build

run:
	docker-compose up

stop:
	docker-compose down

clean:
	docker-compose down --volumes --remove-orphans