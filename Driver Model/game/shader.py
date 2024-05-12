import pygame
from pygame.locals import *

class shader:
    def __init__(self, surf):

        self.size = surf.get_size()
        self.surface = surf                  #

        self.fog = pygame.Surface(self.size) # fog is pygame Surface

    def setup(self, color):
        self.bg = color

    def render(self, *args):
        self.fog.fill(self.bg)

        #print(args)
        for l in args: # l == light object
            l.render(self.fog, l.pos)

        self.surface.blit(self.fog, (0, 0), special_flags=BLEND_MULT)