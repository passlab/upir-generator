      program  main
      implicit none

      integer i, n
      real x(100), y(100), a
      n = 100
!$omp parallel do
      do i = 1, n
        y(i) = y(i) + a * x(i)
      enddo  

      end
