from ctypes.wintypes import RGB
from PyRT_Common import *
from random import randint

from PyRT_Core import Lambertian


# -------------------------------------------------Integrator Classes
# the integrators also act like a scene class in that-
# it stores all the primitives that are to be ray traced.
class Integrator(ABC):
    # Initializer - creates object list
    def __init__(self, filename_, experiment_name=''):
        # self.primitives = []
        self.filename = filename_ + experiment_name
        # self.env_map = None  # not initialized
        self.scene = None

    @abstractmethod
    def compute_color(self, ray):
        pass

    # def add_environment_map(self, env_map_path):
    #    self.env_map = EnvironmentMap(env_map_path)
    def add_scene(self, scene):
        self.scene = scene

    def get_filename(self):
        return self.filename

    # Simple render loop: launches 1 ray per pixel
    def render(self):
        # YOU MUST CHANGE THIS METHOD IN ASSIGNMENTS 1.1 and 1.2:
        cam = self.scene.camera  # camera object
        #ray = Ray()
        print('Rendering Image: ' + self.get_filename())
        print(cam.width, cam.height)
        for x in range(0, cam.width):
            for y in range(0, cam.height):
                ray = Ray(direction=cam.get_direction(x, y))
                pixel = self.compute_color(ray) #RGBColor(x/cam.width, y/cam.height, 0)
                self.scene.set_pixel(pixel, x, y)  # save pixel to pixel array
            progress = (x / cam.width) * 100
            print('\r\tProgress: ' + str(progress) + '%', end='')
        # save image to file
        print('\r\tProgress: 100% \n\t', end='')
        full_filename = self.get_filename()
        self.scene.save_image(full_filename)


class LazyIntegrator(Integrator):
    def __init__(self, filename_):
        super().__init__(filename_ + '_Intersection')

    def compute_color(self, ray):
        return BLACK


class IntersectionIntegrator(Integrator):

    def __init__(self, filename_):
        super().__init__(filename_ + '_Intersection')

    def compute_color(self, ray):
        # ASSIGNMENT 1.2: PUT YOUR CODE HERE
        if self.scene.any_hit(ray):
            return RGBColor(255, 0, 0)
        else:
            return RGBColor(0, 0, 0)


class DepthIntegrator(Integrator):

    def __init__(self, filename_, max_depth_=10):
        super().__init__(filename_ + '_Depth')
        self.max_depth = max_depth_

    def compute_color(self, ray):
        # ASSIGNMENT 1.3: PUT YOUR CODE HERE
        hit_data = self.scene.closest_hit(ray)
        color = RGBColor(0, 0, 0)
        if hit_data.has_hit:
            hit_distance = hit_data.hit_distance
            color_comp = min(1 - (hit_distance/self.max_depth), 1)
            color = RGBColor(color_comp, color_comp, color_comp)
        return color


class NormalIntegrator(Integrator):

    def __init__(self, filename_):
        super().__init__(filename_ + '_Normal')

    def compute_color(self, ray):
        # ASSIGNMENT 1.3: PUT YOUR CODE HERE
        hit_data = self.scene.closest_hit(ray)
        color = RGBColor(0, 0, 0)
        if hit_data.has_hit:
            normal = hit_data.normal
            color = (normal + Vector3D(1, 1, 1))/2
            color = RGBColor(color.x, color.y, color.z)
        return color


class PhongIntegrator(Integrator):

    def __init__(self, filename_):
        super().__init__(filename_ + '_Phong')

    def compute_color(self, ray):
        # ASSIGNMENT 1.4: PUT YOUR CODE HERE
        color = RGBColor(0, 0, 0)
        
        hit_data = self.scene.closest_hit(ray)
        
        if hit_data.has_hit:

            normal = hit_data.normal
            point = hit_data.hit_point
            primitive_idx = hit_data.primitive_index
            primitive = self.scene.object_list[primitive_idx]

            for light_source in self.scene.pointLights:                
                intensity = light_source.intensity
                light_pos = light_source.pos

                x = np.array([light_pos.x, light_pos.y, light_pos.z])
                y = np.array([point.x, point.y, point.z])
                distance = np.linalg.norm(x - y)
                
                light_vector = (light_pos - point)/distance

                ray2 = Ray(point, light_vector, distance)
                if self.scene.any_hit(ray2):
                    continue

                color += primitive.BRDF.get_value(light_vector, 0, normal).multiply(intensity/(distance**2))

            color += self.scene.i_a.multiply(primitive.BRDF.kd)

        return color


class CMCIntegrator(Integrator):  # Classic Monte Carlo Integrator

    def __init__(self, n, filename_, experiment_name=''):
        filename_mc = filename_ + '_MC_' + str(n) + '_samples' + experiment_name
        super().__init__(filename_mc)
        self.n_samples = n

    def compute_color(self, ray):
        pass


class BayesianMonteCarloIntegrator(Integrator):
    def __init__(self, n, myGP, filename_, experiment_name=''):
        filename_bmc = filename_ + '_BMC_' + str(n) + '_samples' + experiment_name
        super().__init__(filename_bmc)
        self.n_samples = n
        self.myGP = myGP

    def compute_color(self, ray):
        pass
