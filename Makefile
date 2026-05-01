up:
	docker compose -f docker/docker-compose.yml up -d
	cd E:\langfuse && docker compose up -d

down:
	docker compose -f docker/docker-compose.yml down
	cd E:\langfuse && docker compose down

logs:
	docker compose -f docker/docker-compose.yml logs -f