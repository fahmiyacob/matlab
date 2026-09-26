# Results — Dhia_Problem3_Inclined_New_Formula

Model dijalankan mengikut arahan dalam `PROMPT.docx`. MATLAB/Octave tiada pada mesin ini,
jadi program `Dhia_Problem3_Inclined_New_Formula.m.txt` dipindahkan 1:1 ke Python
(`scipy.integrate.solve_bvp`, kaedah kolokasi yang setara dengan `bvp4c`, RelTol 1e-8).
Kod Python ada dalam folder `python_port/` (model dalam `dusty_bvp.py`).

**Semakan kesahihan port:**
- Pada parameter asal fail .m, kedua-dua initial guess menumpu dengan kemas.
- Tugasan 1 dengan sigma1=0 memberi f''(0) = -0.99977 ≈ -1, iaitu nilai eksak Crane
  untuk stretching sheet klasik (beza kecil datang dari Gr=0.001 yang kekal aktif).
- Gr=1 dengan X=pi/2 (cos X = 0) menghasilkan semula lengkung Gr=0.001 dengan tepat.

## Tugasan 1 — Jadual sigma1 (`task1_table_sigma1.csv`)

Parameter: lambda=1, phi1=phi2=Bv=BT=s=X=0, Pr=6.2, Ec=1, m=1, epsilon=1.
Parameter yang TIDAK dinyatakan dalam PROMPT dikekalkan pada nilai asal fail .m:
**sigma2=0.2, Gr=0.001, L=1**. Nilai disahkan menumpu sehingga etaMax=30.

| sigma1 | f''(0)       | F'(0) | -t'(0)      | tp(0) |
|--------|--------------|-------|-------------|-------|
| 0      | -0.999770355 | 0     | 1.405296308 | 0     |
| 0.1    | -0.871889376 | 0     | 1.487037403 | 0     |
| 0.2    | -0.776208327 | 0     | 1.536456760 | 0     |
| 0.5    | -0.591067377 | 0     | 1.597089337 | 0     |
| 2.0    | -0.283903629 | 0     | 1.538354886 | 0     |
| 5.0    | -0.144784538 | 0     | 1.369498066 | 0     |

- **Tiada dual solution** pada kes ini — lambda=1 ialah kes stretching; dual solution
  lazimnya wujud untuk shrinking (lambda < 0). Kedua-dua initial guess (OdeInit1 #6 dan
  OdeInit2 #7) menumpu kepada penyelesaian yang sama.
- Dengan Bv=BT=0, fasa habuk terlepas-gandingan sepenuhnya, maka F'(0)=0 dan tp(0)=0 tepat.
- Nota: pada sigma1=2.0 wujud satu penyelesaian matematik kedua yang terpencil
  (f''(0)=-0.4687, -t'(0)=27.14) tetapi t(0)=-4.4 < 0 dan ia tidak boleh diteruskan
  ke nilai sigma1 jiran — ia resonans eigen terma, bukan dual solution fizikal.

## Tugasan 2 — f''(0) dan -t'(0) melawan lambda (X = 0, pi/4, pi/2)

Parameter: phi1=phi2=0.1, Bv=BT=0.5, s=1, Pr=6.2, Ec=1, m=1, epsilon=1
(sigma1=sigma2=0.2, Gr=0.001, L=1 kekal nilai asal fail).

- Fail: `fig_fpp0_vs_lambda.png`, `fig_mtp0_vs_lambda.png`
  (data: `task2_curve_X0.csv`, `task2_curve_Xpi4.csv`, `task2_curve_Xpi2.csv`;
  lajur t'(0) — ambil negatif untuk -t'(0))
- Garis penuh = first solution, garis putus = second solution (dual solution wujud).
- Nilai kritikal: **lambda_c = -0.44928 (X=0), -0.44922 (X=pi/4), -0.44907 (X=pi/2)**.
- **Ketiga-tiga lengkung X hampir bertindih** kerana X hanya masuk melalui sebutan
  Gr·A8·t·cos(X) dan Gr=0.001 dalam program terlalu kecil. Untuk melihat kesan X,
  Gr perlu O(1) — lihat rajah tambahan `fig_*_Gr1.png`: dengan Gr=1, lengkung X=0,
  pi/4, pi/2 berpisah dengan ketara, tetapi sebutan lesapan likat (Ec=1) menyebabkan
  suhu lari (thermal runaway) sebelum fold ditemui bagi X=0 dan pi/4.
- Second solution wujud dalam julat sempit lambda_c < lambda < ~-0.39; melepasi itu
  penyelesaian kedua mengalami resonans terma (-t'(0) menuju tak terhingga).

## Tugasan 3 — Profil halaju & suhu (dual solution, lambda = -0.42, X = 0)

- `fig_velocity_profile.png` (f'(eta)), `fig_temperature_profile.png` (t(eta)),
  `fig_dust_velocity_profile.png` (F'(eta)), `fig_dust_temperature_profile.png` (tp(eta)).
- Data penuh: `task3_profile_first_solution.csv`, `task3_profile_second_solution.csv`.
- First solution: f''(0)=0.326205, -t'(0)=1.557746; second: f''(0)=0.216472, -t'(0)=1.358579.
- Semua syarat sempadan `OdeBC` dipenuhi pada aras 1e-17 (residual dicetak oleh
  `python_port/task3_profiles.py`), dan semua profil menyusut ke 0 pada eta besar.

## Untuk ulang dalam MATLAB

Tugasan 1: dalam fail .m set `lambda=1; phi1=0; phi2=0; Bv=0; BT=0; s=0; X=0; Pr=6.2;
Eckert=1; m=1; epsilon=1;` dan ubah `sigma1` mengikut jadual. Guna `etaMax1 >= 15`
(nilai 6 memberi ralat pemotongan domain yang ketara, terutama sigma1 besar).
Tugasan 2: kekalkan parameter di atas dengan `phi1=phi2=0.1; Bv=BT=0.5; s=1;` dan
loop `lambda` dari 1 turun ke -0.4493; second solution hanya dalam
[-0.4493, -0.39] — perlukan initial guess dari penyelesaian jiran (continuation),
guess tetap `OdeInit2` sukar menumpu di situ.
