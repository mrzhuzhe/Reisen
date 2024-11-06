//
// Created by goksu on 2/25/20.
//

#include <fstream>
#include "Scene.hpp"
#include "Renderer.hpp"
#include "pthread.h"

inline float deg2rad(const float& deg) { return deg * M_PI / 180.0; }

const float EPSILON = 0.00001;

struct  thread_info
{
    pthread_t thread_id;
    int thread_num;
    float imageAspectRatio;
    float scale;
    const Scene* scene;
    size_t num_threads;
    int spp;
    std::vector<Vector3f>* framebuffer;
    Vector3f eye_pos;
};

void *thread_fn(void *arg){
    thread_info *_info = (thread_info*)arg;
    unsigned thread_num = _info->thread_num;
    float imageAspectRatio = _info->imageAspectRatio;
    float scale = _info->scale;
    int spp = _info->spp;
    size_t num_threads = _info->num_threads;
    const Scene* scene = _info->scene;
    std::vector<Vector3f>* framebuffer = _info->framebuffer;
   
    Vector3f eye_pos(278, 273, -800);
    unsigned stride = scene->height / num_threads;
    unsigned start = thread_num * stride;
    int m = start*scene->width;
    
    for (uint32_t j = start; j < start + stride; ++j) {
        for (uint32_t i = 0; i < scene->width; ++i) {
            // generate primary ray direction
            float x = (2 * (i + 0.5) / (float)scene->width - 1) *
                      imageAspectRatio * scale;
            float y = (1 - 2 * (j + 0.5) / (float)scene->height) * scale;

            Vector3f dir = normalize(Vector3f(-x, y, 1));
            for (int k = 0; k < spp; k++){
                (*framebuffer)[m] += scene->castRay(Ray(eye_pos, dir), 0) / spp;  
            }
            m+=1;
        };
    }
    return NULL;
}

// The main render function. This where we iterate over all pixels in the image,
// generate primary rays and cast these rays into the scene. The content of the
// framebuffer is saved to a file.
void Renderer::Render(const Scene& scene, int spp)
{
    std::vector<Vector3f> framebuffer(scene.width * scene.height);
    float scale = tan(deg2rad(scene.fov * 0.5));
    float imageAspectRatio = scene.width / (float)scene.height;
    Vector3f eye_pos(278, 273, -800);

    // change the spp value to change sample ammount
    std::cout << "SPP: " << spp << "\n";
    
    size_t num_threads = 32;
    thread_info tinfo[num_threads];

    for (int tnum=0; tnum<num_threads; tnum++){
        tinfo[tnum].thread_num = tnum;
        tinfo[tnum].scene = &scene;
        tinfo[tnum].imageAspectRatio = imageAspectRatio;
        tinfo[tnum].scale = scale;
        tinfo[tnum].spp = spp;
        tinfo[tnum].num_threads = num_threads;
        tinfo[tnum].framebuffer = &framebuffer;
        tinfo[tnum].eye_pos = eye_pos;
        pthread_create(&tinfo[tnum].thread_id, NULL, &thread_fn, &tinfo[tnum]);
    }
    
    for (int tnum=0; tnum<num_threads; tnum++){
        pthread_join(tinfo[tnum].thread_id, NULL);
    }

    UpdateProgress(1.f);

    // save framebuffer to file
    FILE* fp = fopen("binary.ppm", "wb");
    (void)fprintf(fp, "P6\n%d %d\n255\n", scene.width, scene.height);
    for (auto i = 0; i < scene.height * scene.width; ++i) {
        static unsigned char color[3];
        color[0] = (unsigned char)(255 * std::pow(clamp(0, 1, framebuffer[i].x), 0.6f));
        color[1] = (unsigned char)(255 * std::pow(clamp(0, 1, framebuffer[i].y), 0.6f));
        color[2] = (unsigned char)(255 * std::pow(clamp(0, 1, framebuffer[i].z), 0.6f));
        fwrite(color, 1, 3, fp);
    }
    fclose(fp);    
}
