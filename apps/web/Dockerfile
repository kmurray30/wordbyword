# Railway's documented pattern for a Vite/React static site: build with node,
# serve the `dist` output with Caddy on Railway's assigned $PORT.
FROM node:20-alpine AS build
WORKDIR /app
COPY package*.json .npmrc ./
RUN npm ci
COPY . .

# Vite inlines VITE_* env vars at build time, so this has to be a build ARG
# (Railway populates ARGs from the service's variables of the same name),
# not a runtime env var.
ARG VITE_API_BASE_URL
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL
RUN npm run build

FROM caddy:2-alpine
WORKDIR /srv
COPY --from=build /app/dist ./dist
COPY Caddyfile /etc/caddy/Caddyfile
