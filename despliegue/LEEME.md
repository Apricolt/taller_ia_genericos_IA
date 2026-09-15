# Despliegue: frontend en Vercel, backend en Render

Estado a 14/09/2026: **preparado, no desplegado**. El proyecto web todavia no
existe (`web/pages` esta vacio), asi que aqui solo estan las piezas listas para
activarse. Ningun archivo de esta carpeta esta en uso: las herramientas los
buscan en la raiz del proyecto, no aqui dentro.

## Por que dos servicios y no uno

Vercel no ejecuta procesos Python largos: esta pensado para archivos estaticos
y funciones que responden en segundos. Una corrida de algoritmo genetico no
encaja ahi. Render si corre un proceso normal y continuo.

Reparto entonces:

- **Vercel** -> la interfaz (HTML/JS, o React/Vite/Next). Rapido y gratis.
- **Render** -> la API Python que llama a `ga/` y `problemas/`.

## Ya hecho

- Vercel CLI **59.17.0** instalado (`npm install -g vercel`).
- Comprobado que `ga/estadisticas.py` ya usa `matplotlib.use("Agg")`, el modo
  sin ventana que un servidor necesita. No hay que cambiar nada del codigo.
- Dependencias reales detectadas: numpy, pandas, matplotlib.

## Falta (lo tienes que hacer tu)

1. **Iniciar sesion en Vercel.** Abre una terminal y ejecuta:

       vercel login

   Se abre el navegador para autenticarte. Comprueba despues con `vercel whoami`.

2. **Convertir esto en repositorio git.** `Taller1_Geneticos` no lo es todavia.
   Vercel puede desplegar sin repo, pero **Render no**: solo despliega desde
   GitHub/GitLab/Bitbucket.

       git init
       git add .
       git commit -m "Taller 1: algoritmos geneticos"

   Luego crea el repo en GitHub y conectalo como `origin`.

3. **Escribir el backend**, por ejemplo `api.py` en la raiz, con Flask
   exponiendo lo que ya hace `experimentos/`.

## Cuando el proyecto este listo

### Backend en Render

    cp despliegue/requirements-servidor.txt requirements.txt
    cp despliegue/render.yaml render.yaml

Ajusta en `render.yaml` el `startCommand`: `api:app` debe coincidir con tu
archivo y tu variable Flask. Luego en render.com: **New > Blueprint** y apunta
al repositorio.

Tres cosas del plan gratuito que conviene saber de antemano:

- El servicio **se duerme tras ~15 min sin trafico**. La siguiente peticion
  tarda ~50 s en responder mientras vuelve a arrancar. No es un fallo.
- La CPU es limitada: una corrida genetica pesada puede tardar bastante mas
  que en tu portatil. Si pasa de ~2 min, no la ejecutes dentro de la peticion
  HTTP; lanzala en segundo plano y que el frontend consulte el resultado.
- El disco no es permanente: los PNG y CSV que escriba el servidor
  desaparecen en cada redespliegue. Devuelvelos en la respuesta en vez de
  confiar en que sigan ahi.

### Frontend en Vercel

    cp despliegue/vercel.json vercel.json
    cp despliegue/.vercelignore .vercelignore

En `vercel.json` hay que tocar dos cosas:

- `buildCommand` y `outputDirectory` segun lo que uses. Con Vite son
  `npm run build` y `dist` (lo que esta puesto); con Next.js borra ambas
  lineas y pon `"framework": "nextjs"`.
- En `rewrites`, cambia `CAMBIAR-POR-TU-SERVICIO.onrender.com` por la URL real
  que te de Render. Ese rewrite hace que el navegador vea la API como
  `/api/...` del mismo dominio, y asi **no hay problemas de CORS**.

Despliegue:

    vercel          # entorno de vista previa, para revisar
    vercel --prod   # produccion

La primera vez pregunta a que proyecto asociar la carpeta; guarda la respuesta
en `.vercel/` y ya no vuelve a preguntar.

## Comprobacion rapida antes de dar algo por desplegado

- `vercel whoami` responde con tu usuario.
- La URL de Render responde a `/api/salud` con 200.
- El frontend desplegado llama a `/api/...` y recibe datos, no un 404.
