# NRG-RadianceProbesForGlobalIlluminationInVolumeRendering

## Radiance probes for global illumination in volume rendering
Accurate and efficient computation of global illumination is a significant challenge in volume rendering. The
appearance of materials is heavily influenced by the surrounding environment, making the quality of rendering
depend on effective environment sampling. While path tracing can produce accurate results, it is often too slow for
real-time execution. A common practical solution is to precompute radiance at fixed points in space known as
radiance probes or light probes and use these values during rendering to avoid costly computation for every light ray.
In this seminar, you will explore the radiance probe method and implement it in a prototype application. You will
evaluate the results both qualitatively and quantitatively and compare them to path tracing.


## Building

The project uses webpack and node to build, after cloning the repo run:

```
npm install
npm run serve
```

Then point your browser to `localhost:8080`!


### References:

`
[1] R. Khlebnikov, P. Voglreiter, M. Steinberger, B. Kainz, D. Schmalstieg, “Parallel Irradiance Caching for
Interactive Monte‐Carlo Direct Volume Rendering,” Computer Graphics Forum, vol. 33, no. 3, 2014, doi:
10.1111/cgf.12362
`

# WebGPU Volume Pathtracer

A volume raycaster using WebGPU. Try it out [online!](https://www.willusher.io/webgpu-volume-pathtracer/)

![bonsai](https://user-images.githubusercontent.com/1522476/177204612-f305c151-6e41-44bb-ba4c-1fd334beec60.png)
