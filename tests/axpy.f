      program  main
      implicit none

      integer i, n
      real x(100), y(100), a
      n = 100
!$omp parallel do num_threads(6)
      do i = 1, n
        y(i) = y(i) + a * x(i)
      enddo  

      end
